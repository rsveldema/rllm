#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* See feature_test_macros(7) */
#endif

#include <ifaddrs.h>
#include <netdb.h>
#include <sys/mman.h>

#include <thread>

#include "IOUring.hpp"
#include "ProbeUringFeatures.hpp"
#include "SocketImpl.hpp"
#include "WorkItem.hpp"

namespace iuring {
namespace {
// global instance to help with signal handler callback
std::shared_ptr<IOUring> s_self;
} // namespace

std::shared_ptr<IOUringInterface> IOUringInterface::create_impl(
    logging::ILogger &logger, NetworkAdapter &adapter) {
  return IOUring::create(logger, adapter);
}

std::shared_ptr<IOUring> IOUring::create(
    logging::ILogger &logger, NetworkAdapter &adapter, size_t queue_size) {
  /** make_shared<> does not work with private ctors
   * so we inherit from it with a public ctor
   * which can be make_shared<>.
   */
  class EnableShared : public IOUring {
  public:
    EnableShared(logging::ILogger &logger, NetworkAdapter &adapter,
                 size_t queue_size)
        : IOUring(logger, adapter, queue_size) {
    }
  };

  return std::make_shared<EnableShared>(logger, adapter, queue_size);
}

IOUring::IOUring(
    logging::ILogger &logger, NetworkAdapter &adapter, size_t queue_size)
    : m_logger(logger), m_queue_size(queue_size), m_adapter(adapter), m_pool(logger) {
}

IOUring::~IOUring() {
  io_uring_queue_exit(&m_ring);
}

error::Error IOUring::init() {
  init_ring();

  probe_features();

  auto ret = setup_buffer_pool();
  m_initialized = true;
  return ret;
}

void IOUring::init_ring() {
  if (false) {
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    params.cq_entries = QD * 8;
    params.flags =
        // IORING_SETUP_IOPOLL | // only for storage
        IORING_SETUP_SUBMIT_ALL |    //
        IORING_SETUP_COOP_TASKRUN |  //
        IORING_SETUP_SINGLE_ISSUER | //
        IORING_SETUP_DEFER_TASKRUN | //
        IORING_SETUP_CQSIZE |        //
        0;

    if (const auto ret = io_uring_queue_init_params(QD, &m_ring, &params);
        ret < 0) {
      LOG_ERROR(get_logger(),
                "queue_init failed: {}\n"
                "NB: This requires a kernel version >= 6.0",
                strerror(-ret));
      abort();
    }

    if (1) {
      auto ret = io_uring_register_ring_fd(&m_ring);
      if (ret < 0) {
        LOG_ERROR(get_logger(), "register_ring_fd: {}", strerror(-ret));
        abort();
      }

      ret = io_uring_close_ring_fd(&m_ring);
      if (ret < 0) {
        LOG_ERROR(get_logger(), "close_ring_fd: {}\n", strerror(-ret));
        abort();
      }
    }
  } else {
    if (const auto ret = io_uring_queue_init(m_queue_size, &m_ring, 0);
        ret != 0) {
      LOG_ERROR(
          get_logger(), "io_uring_queue_init: {}\n", strerror(-ret));
      abort();
    }
  }

  io_uring_ring_dontfork(&m_ring);
}

error::Error IOUring::setup_buffer_pool() {
  buf_ring_size = (sizeof(io_uring_buf) + buffer_size()) * BUFFERS;
  void *mapped = mmap(NULL, buf_ring_size, PROT_READ | PROT_WRITE,
                      MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
  if (mapped == MAP_FAILED) {
    LOG_ERROR(get_logger(), "buf_ring mmap: {}\n", strerror(errno));
    return error::Error::MMAP_FAILED;
  }
  buf_ring = (struct io_uring_buf_ring *)mapped;

  io_uring_buf_ring_init(buf_ring);

  memset(&m_reg, 0, sizeof(m_reg));
  m_reg.ring_addr = (unsigned long)buf_ring;
  m_reg.ring_entries = BUFFERS;
  m_reg.bgid = 0;
  m_reg.flags = 0;

  buffer_base =
      (unsigned char *)buf_ring + sizeof(struct io_uring_buf) * BUFFERS;

  const auto ret = io_uring_register_buf_ring(&m_ring, &m_reg, 0);
  if (ret) {
    LOG_ERROR(get_logger(),
              "buf_ring init failed: {}\n"
              "NB This requires a kernel version >= 6.0\n",
              strerror(-ret));
    return error::errno_to_error(-ret);
  }

  for (auto i = 0u; i < BUFFERS; i++) {
    io_uring_buf_ring_add(buf_ring, get_buffer(i), buffer_size(), i,
                          io_uring_buf_ring_mask(BUFFERS), i);
  }
  io_uring_buf_ring_advance(buf_ring, BUFFERS / 2);

  for (int i = BUFFERS / 2; i < BUFFERS; i++) {
    m_free_send_ids.push(i);
  }

  return error::Error::OK;
}

void IOUring::recycle_buffer(int idx) {
  io_uring_buf_ring_add(buf_ring, get_buffer(idx), buffer_size(), idx,
                        io_uring_buf_ring_mask(BUFFERS), 0);
  io_uring_buf_ring_advance(buf_ring, 1);
}

void IOUring::probe_features() {
  ProbeUringFeatures probe(&m_ring, get_logger());
  assert(probe.supports(UringFeature::IORING_OP_ACCEPT));
#if SUPPORT_LISTEN_IN_LIBURING
  assert(probe.supports(UringFeature::IORING_OP_LISTEN));
#endif
  assert(probe.supports(UringFeature::IORING_OP_RECV));
  assert(probe.supports(UringFeature::IORING_OP_RECVMSG));
  assert(probe.supports(UringFeature::IORING_OP_SEND));
  assert(probe.supports(UringFeature::IORING_OP_SENDMSG));
  assert(probe.supports(UringFeature::IORING_OP_CLOSE));
  assert(probe.supports(UringFeature::IORING_OP_CONNECT));
}

void IOUring::submit_all_requests() {
  // fprintf(stderr, "SUBMIT REQUEST!\n");
  // unsigned wait_nr = 1;
  // const auto ret = io_uring_submit_and_wait(&m_ring, wait_nr);
  const auto ret = io_uring_submit(&m_ring);
  if (ret < 0) {
    LOG_ERROR(get_logger(), "failed to submit sqe: {}", strerror(-ret));
  } else {
    // fprintf(stderr, "{} jobs submitted\n", ret);
  }
}

io_uring_sqe *IOUring::get_sqe() {
  auto *sqe = io_uring_get_sqe(&m_ring);
  if (!sqe) {
    LOG_ERROR(get_logger(), "sqe entry is NULL from get-sqe\n");
    submit_all_requests();
    sqe = io_uring_get_sqe(&m_ring);
  }

  if (!sqe) {
    LOG_ERROR(get_logger(), "sqe entry is NULL from get-sqe\n");
    abort();
  }
  return sqe;
}

void IOUring::submit(IWorkItem &_item) {
  auto &item = dynamic_cast<WorkItem &>(_item);
  auto *sqe = get_sqe();
  io_uring_sqe_set_data(sqe, (void *)item.m_id);

  switch (item.get_type()) {
  default:
    LOG_ERROR(get_logger(), "INTERNAL ERROR: unhandled work item type\n");
    abort();

  case WorkItem::Type::CLOSE:
    io_uring_prep_close(sqe, item.get_socket()->get_fd());
    break;

  case WorkItem::Type::ACCEPT: {
    int flags = 0;
    // flags |= IOSQE_BUFFER_SELECT;

    LOG_DEBUG(
        get_logger(), "accept on socket {}", item.get_socket()->get_fd());

    item.m_accept_sock_len = 0;
    io_uring_prep_accept(sqe, item.get_socket()->get_fd(),
                         (struct sockaddr *)&item.m_buffer_for_uring,
                         &item.m_accept_sock_len, flags);
    break;
  }

  case WorkItem::Type::CONNECT: {
    assert(item.m_connect_sock_len > 0);
    const auto fd = item.get_socket()->get_fd();

    sockaddr_in *sa = (sockaddr_in *)&item.m_buffer_for_uring;

    assert(item.m_connect_sock_len == sizeof(*sa));

    LOG_DEBUG(get_logger(), "prep-connect: fd={} (port {})", fd,
              htons(sa->sin_port));

    io_uring_prep_connect(sqe, fd,
                          (struct sockaddr *)&item.m_buffer_for_uring,
                          item.m_connect_sock_len);
    if (item.next_request_should_wait_for_this_request()) {
      sqe->flags |= IOSQE_IO_LINK;
    }
    break;
  }

  case WorkItem::Type::RECV: {
    if (item.is_stream()) {
      int flags = 0;
      LOG_DEBUG(
          get_logger(), " register rcv: {}", item.get_socket()->get_fd());
      io_uring_prep_recv(sqe, item.get_socket()->get_fd(),
                         nullptr, // buffer selected automatically from buffer queue
                         buffer_size(), flags);
    } else {
      // fprintf(stderr, "RECV ---- submit: {}\n", idx);
      memset(&item.m_msg, 0, sizeof(item.m_msg));

      item.m_msg.msg_name = &item.m_buffer_for_uring;
      item.m_msg.msg_namelen = sizeof(item.m_buffer_for_uring);
      item.m_msg.msg_iov = item.m_msg_iov->data();
      item.m_msg.msg_iovlen = item.m_msg_iov->size();

      assert(item.m_msg.msg_iovlen == 1);

      // fprintf(stderr, "msg_name = %p, p = %p\n",  item.m_msg.msg_name,
      // item.m_msg.msg_iov);

      item.m_msg.msg_iov->iov_base =
          nullptr; // selects a buffer automatically from buffer-queue

      io_uring_prep_recvmsg_multishot(
          sqe, item.get_socket()->get_fd(), &item.m_msg, MSG_TRUNC);
    }

    sqe->flags |= IOSQE_BUFFER_SELECT;
    sqe->buf_group = 0;
    break;
  }

  case WorkItem::Type::SEND_STREAM_DATA: {
    const auto fd = item.get_socket()->get_fd();

    assert(item.is_stream());
    int flags = 0;
    const auto &sp = item.get_raw_send_packet();

    LOG_DEBUG(get_logger(), "sending {} bytes ({})", sp.size(),
              (char *)sp.data());
    io_uring_prep_send(sqe, fd, sp.data(), sp.size(), flags);

    if (item.next_request_should_wait_for_this_request()) {
      sqe->flags |= IOSQE_IO_LINK;
    }
    break;
  }

  case WorkItem::Type::SEND_WORKPACKET: {
    const auto fd = item.get_socket()->get_fd();

    assert(!item.is_stream());
    int flags = 0;
    LOG_DEBUG(get_logger(), "SEND ---- submit: {}", fd);
    item.init_send_msg();
    io_uring_prep_sendmsg(sqe, fd, &item.m_msg, flags);

    // sqe->flags |= IOSQE_FIXED_FILE;
    // sqe->flags |= IOSQE_BUFFER_SELECT;

    if (item.next_request_should_wait_for_this_request()) {
      sqe->flags |= IOSQE_IO_LINK;
    }
    break;
  }
  }

  submit_all_requests();
}

void WorkItem::init_send_msg() {
  assert(m_work_type == WorkItem::Type::SEND_WORKPACKET);

  m_msg.msg_iov = m_msg_iov->data();
  m_msg.msg_iovlen = m_msg_iov->size();
  assert(m_msg.msg_iovlen == 1);

  m_msg.msg_iov[0].iov_base = (void *)m_send_packet.data();
  m_msg.msg_iov[0].iov_len = m_send_packet.size();

  {
    const uint8_t congestion_notification = 0;
    const uint8_t tos =
        static_cast<std::underlying_type_t<dscp_t>>(m_params.dscp) << 2 |
        congestion_notification;
    const int32_t ttl =
        static_cast<std::underlying_type_t<timetolive_t>>(m_params.ttl);
    assert(ttl > 0 && ttl < 256);

    m_control.fill(0);

    m_msg.msg_control = m_control.data();
    m_msg.msg_controllen =
        CMSG_SPACE(sizeof(tos)) + CMSG_SPACE(sizeof(ttl));
    assert(m_msg.msg_controllen < m_control.size());

    auto *cmsgptr = CMSG_FIRSTHDR(&m_msg);
    assert(cmsgptr);
    cmsgptr->cmsg_level = IPPROTO_IP;
    cmsgptr->cmsg_type = IP_TOS;
    cmsgptr->cmsg_len = CMSG_LEN(sizeof(tos));
    memcpy(CMSG_DATA(cmsgptr), &tos, sizeof(tos));

    cmsgptr = CMSG_NXTHDR(&m_msg, cmsgptr);
    assert(cmsgptr);
    cmsgptr->cmsg_level = IPPROTO_IP;
    cmsgptr->cmsg_type = IP_TTL;
    cmsgptr->cmsg_len = CMSG_LEN(sizeof(ttl));
    memcpy(CMSG_DATA(cmsgptr), &ttl, sizeof(ttl));
  }

  m_msg.msg_flags = 0;

  m_sa = m_params.destination_address;

  if (const auto *a = m_sa.get_ipv4()) {
    m_msg.msg_name = (void *)a;
    m_msg.msg_namelen = sizeof(*a);
  } else if (const auto *a = m_sa.get_ipv6()) {
    m_msg.msg_name = (void *)a;
    m_msg.msg_namelen = sizeof(*a);
  } else {
    abort();
  }
  // LOG_DEBUG(get_logger(), "submitting send: {} bytes\n", size);
}

void IOUring::call_close_callback(
    std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe) {
  const int status = cqe->res;
  LOG_DEBUG(get_logger(), "=======> CLOSE CALLBACK: {}", cqe->res);
  work_item->call_close_callback(status);
}

void IOUring::call_send_callback(
    std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe) {
  LOG_DEBUG(get_logger(), "=======> SEND CALLBACK: {}", cqe->res);
  if (cqe->res < 0) {
    LOG_ERROR(get_logger(), "recv cqe bad res {}", cqe->res);
    if (cqe->res == -EFAULT || cqe->res == -EINVAL) {
      LOG_ERROR(
          get_logger(), "NB: This requires a kernel version >= 6.0\n");
      return;
    }
  }

  work_item->call_send_callback(cqe->res);
}

void IOUring::call_connect_callback(
    std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe) {
  LOG_DEBUG(get_logger(), "=======> CONNECT CALLBACK: {}", cqe->res);
  if (cqe->res < 0) {
    LOG_ERROR(get_logger(), "recv cqe bad res {} ({})", cqe->res,
              strerror(-cqe->res));
    if (cqe->res == -EFAULT || cqe->res == -EINVAL) {
      LOG_ERROR(
          get_logger(), "NB: This requires a kernel version >= 6.0\n");
      return;
    }
  }

  const int status = cqe->res;

  sockaddr_in *sa = (sockaddr_in *)&work_item->m_buffer_for_uring;
  iuring::IPAddress addr;
  if (sa->sin_family == AF_INET) {
    addr = IPAddress(*sa);
  } else if (sa->sin_family == AF_INET6) {
    addr = IPAddress(*(sockaddr_in6 *)sa);
  }

  const ConnectResult new_conn(status, addr);

  LOG_DEBUG(get_logger(), "CONN- XQE - res = {}", status);

  work_item->call_connect_callback(new_conn);
}

void IOUring::call_accept_callback(
    std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe) {
  // if (!(cqe->flags & IORING_CQE_F_BUFFER) || cqe->res < 0)
  if (cqe->res < 0) {
    LOG_ERROR(get_logger(), "recv cqe bad res {} ({})", cqe->res,
              strerror(-cqe->res));
    if (cqe->res == -EFAULT || cqe->res == -EINVAL) {
      LOG_ERROR(
          get_logger(), "NB: This requires a kernel version >= 6.0\n");
      return;
    }
  }

  const int fd = cqe->res;

  LOG_DEBUG(get_logger(), " XQE - res = {}", fd);

  const iuring::IPAddress addr(
      work_item->m_buffer_for_uring, work_item->m_accept_sock_len);
  const AcceptResult new_conn(fd, addr);

  work_item->call_accept_callback(new_conn);
}

ReceivePostAction IOUring::call_recv_handler_stream(const uint8_t *buffer,
                                                    std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe) {
  IPAddress source_addr;
  const auto payload_length = cqe->res;

  LOG_DEBUG(get_logger(), "size = {}\n", (int)payload_length);

  ReceivedMessage payload(buffer, payload_length, source_addr);

  return work_item->call_recv_callback(payload);
}

ReceivePostAction IOUring::call_recv_handler_datagram(const uint8_t *buffer,
                                                      std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe) {
  if (!(cqe->flags & IORING_CQE_F_BUFFER)) {
    LOG_ERROR(get_logger(), "recv cqe bad res {}", cqe->res);
    abort();
    return ReceivePostAction::RE_SUBMIT;
  }

  auto *recv_msg_out =
      io_uring_recvmsg_validate((void *)buffer, cqe->res, &work_item->m_msg);
  if (!recv_msg_out) {
    LOG_ERROR(get_logger(), "bad recvmsg - no recv_msg_out\n");

    return ReceivePostAction::RE_SUBMIT;
  }

  if (recv_msg_out->namelen > sizeof(sockaddr_storage)) {
    LOG_ERROR(get_logger(), "truncated name\n");

    return ReceivePostAction::RE_SUBMIT;
  }

  if (recv_msg_out->flags & MSG_TRUNC) {
    const auto r = io_uring_recvmsg_payload_length(
        recv_msg_out, cqe->res, &work_item->m_msg);

    LOG_ERROR(get_logger(), "truncated msg need {} received {}",
              recv_msg_out->payloadlen, r);

    return ReceivePostAction::RE_SUBMIT;
  }

  iuring::IPAddress source_addr;
  switch (recv_msg_out->namelen) {
  case 0:
  case sizeof(sockaddr_in): {
    const auto *tmp = (sockaddr_in *)io_uring_recvmsg_name(recv_msg_out);
    source_addr = IPAddress(*tmp);
    break;
  }

  case sizeof(sockaddr_in6): {
    const auto *tmp = (sockaddr_in6 *)io_uring_recvmsg_name(recv_msg_out);
    source_addr = IPAddress(*tmp);
    break;
  }

  default: {
    LOG_ERROR(get_logger(), "namelen = {}", recv_msg_out->namelen);
    abort();
  }
  }

  const auto payload_length = io_uring_recvmsg_payload_length(
      recv_msg_out, cqe->res, &work_item->m_msg);

  LOG_DEBUG(get_logger(),
            "io_uring: received {} bytes (namelen = {}) from {}", payload_length,
            work_item->m_msg.msg_namelen,
            source_addr.to_human_readable_string().c_str());

  auto *ptr =
      (uint8_t *)io_uring_recvmsg_payload(recv_msg_out, &work_item->m_msg);
  assert(ptr);

  ReceivedMessage payload(ptr, payload_length, source_addr);

  return work_item->call_recv_callback(payload);
}

ReceivePostAction IOUring::call_recv_callback(
    std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe) {
  if (cqe->res < 0) {
    LOG_ERROR(get_logger(), "recv cqe bad res {} ({})", cqe->res,
              strerror(-cqe->res));
    switch (cqe->res) {
    case -EFAULT: {
      LOG_ERROR(get_logger(),
                "NB: This requires a kernel version >= 6.0 (EFAULT)");
      break;
    }

    case -EINVAL: {
      LOG_ERROR(get_logger(),
                "NB: This requires a kernel version >= 6.0 (EINVAL)");
      break;
    }

    case EAI_INTR:
      return ReceivePostAction::RE_SUBMIT;
    }
    abort();
    return ReceivePostAction::RE_SUBMIT;
  }

  const auto idx = cqe->flags >> 16;
  auto *buffer = get_buffer(idx);

  ReceivePostAction ret;
  if (work_item->is_stream()) {
    ret = call_recv_handler_stream(buffer, work_item, cqe);
  } else {
    ret = call_recv_handler_datagram(buffer, work_item, cqe);
  }
  recycle_buffer(idx);
  return ret;
}

void IOUring::call_callback_and_free_work_item_id(io_uring_cqe *cqe) {
  const auto recv_status = cqe->res;
  const auto id = (work_item_id_t)io_uring_cqe_get_data(cqe);

  auto work_item = get_pool().get_work_item(id);
  if (!work_item) {
    LOG_ERROR(get_logger(),
              "no work item {} exists anymore (status {}, flags {}, res = {})",
              id, recv_status, cqe->flags, cqe->res);
    return;
  }
  assert(work_item);

  if (cqe->res == -ENOBUFS) {
    LOG_ERROR(get_logger(),
              "uring ---> ENOBUFS buffer??? -- status: {} ({})", recv_status,
              work_item->get_descr().c_str());
    return;
  }

  if (cqe->flags & IORING_CQE_F_MORE) {
    LOG_DEBUG(get_logger(), "NOTE: more completion events to follow ({})",
              work_item->get_descr().c_str());
    // return;
  }

  switch (work_item->get_type()) {
  case WorkItem::Type::ACCEPT:
    call_accept_callback(work_item, cqe);
    // try accept again:
    submit(*work_item);
    break;

  case WorkItem::Type::CLOSE:
    call_close_callback(work_item, cqe);
    get_pool().free_work_item(id);
    break;

  case WorkItem::Type::RECV: {
    auto ret = call_recv_callback(work_item, cqe);
    switch (ret) {
    case ReceivePostAction::NONE:
      get_pool().free_work_item(id);
      break;
    case ReceivePostAction::RE_SUBMIT:
      submit(*work_item);
      break;
    }
    break;
  }

  case WorkItem::Type::CONNECT:
    call_connect_callback(work_item, cqe);
    get_pool().free_work_item(id);
    break;

  case WorkItem::Type::SEND_STREAM_DATA:
  case WorkItem::Type::SEND_WORKPACKET:
    call_send_callback(work_item, cqe);
    get_pool().free_work_item(id);
    break;

  default:
    assert(false);
  }
}

void IOUring::send_packet(const std::shared_ptr<WorkItem> &work_item) {
  submit(*work_item);
}

error::Error IOUring::poll_completion_queues() {
  if (false) {
    // fprintf(stderr, "waiting for incoming msgs\n");
    const auto ret = io_uring_submit(&m_ring);
    if (ret < 0) {
      perror("failed to io-submit");
      return error::Error::UNKNOWN;
    }
  }

  // no loop around this to get more bandwidth.
  // We are optimizing for latency/reliable execution of this task.

  io_uring_cqe *cqe = nullptr;
  const auto success = io_uring_peek_cqe(&m_ring, &cqe);
  switch (success) {
  case 0:
    // fprintf(stderr, "peek successful!\n");

    call_callback_and_free_work_item_id(cqe);
    io_uring_cq_advance(&m_ring, 1);
    break;

  case -EAGAIN:
    break;

  default:
    LOG_ERROR(get_logger(), "failed: {}\n", strerror(-success));
    abort();
    break;
  }
  return error::Error::OK;
}

void IOUring::submit_accept(
    const std::shared_ptr<ISocket> &socket, accept_callback_func_t handler) {
  assert(socket->get_kind() == SocketKind::SERVER_STREAM_SOCKET);
  assert(m_initialized);
  get_pool().alloc_accept_work_item(
      socket, shared_from_this(), handler, "accept-job");
}

void IOUring::submit_connect(const std::shared_ptr<ISocket> &socket,
                             const IPAddress &target, connect_callback_func_t handler) {
  assert(m_initialized);
  get_pool().alloc_connect_work_item(
      target, socket, shared_from_this(), handler, "connect-job");
}

void IOUring::submit_recv(
    const std::shared_ptr<ISocket> &socket, recv_callback_func_t handler) {
  assert(m_initialized);
  get_pool().alloc_recv_work_item(
      socket, shared_from_this(), handler, "read-from-socket");
}

std::shared_ptr<IWorkItem> IOUring::ackuire_send_workitem(
    const std::shared_ptr<ISocket> &socket) {
  assert(m_initialized);
  auto item = get_pool().alloc_send_work_item(
      socket, shared_from_this(), "write-from-socket");
  return item;
}

void IOUring::submit_close(
    const std::shared_ptr<ISocket> &socket, close_callback_func_t handler) {
  assert(m_initialized);
  get_pool().alloc_close_work_item(
      socket, shared_from_this(), handler, "close-of-socket");
}

void IOUring::sig_notifier_hostname_resolve(sigval_t sv) {
  void *ptr = sv.sival_ptr;

  assert(s_self != nullptr);
  s_self->trigger_hostname_resolve_callbacks(ptr);
}

namespace {
std::vector<IPAddress> convert_addresses(const addrinfo *res) {
  std::vector<IPAddress> addresses;

  for (const auto *rp = res; rp != nullptr; rp = rp->ai_next) {
    if (rp->ai_family == AF_INET) {
      sockaddr_in *sa = (sockaddr_in *)rp->ai_addr;
      addresses.push_back(IPAddress(*sa));
    } else if (rp->ai_family == AF_INET6) {
      sockaddr_in6 *sa = (sockaddr_in6 *)rp->ai_addr;
      addresses.push_back(IPAddress(*sa));
    }
  }

  return addresses;
}
} // namespace

void IOUring::trigger_hostname_resolve_callbacks(void *ptr) {
  for (auto &it : m_hostname_DNS_requests) {
    if (!it.request) {
      LOG_INFO(get_logger(),
               "hostname resolve request already handled for {}",
               it.hostname.c_str());
      continue;
    }

    if (it.request != ptr) {
      LOG_INFO(get_logger(),
               "hostname resolve request not matching for {}",
               it.hostname.c_str());
      continue;
    }

    resolve_hostname_arg_t result;

    if (it.request->ar_result == nullptr) {
      LOG_ERROR(
          get_logger(), "hostname resolve failed for {}", it.hostname);
      result = std::unexpected(error::Error::HOSTNAME_RESOLVE_FAILED);
    } else {
      result = convert_addresses(it.request->ar_result);
      freeaddrinfo(it.request->ar_result);
      it.request->ar_result = nullptr;

      free(it.request);
      it.request = nullptr;
    }

    for (auto &handler : it.handlers) {
      handler(result);
    }
  }
}

void IOUring::resolve_hostname(const std::string &hostname,
                               const resolve_hostname_callback_func_t &handler) {
  // if hostname is an IP address, return it directly
  if (auto addr_res = IPAddress::parse(hostname)) {
    LOG_INFO(get_logger(), "IP address conversion suffices: {}, its {}", hostname.c_str(), addr_res.value().to_human_readable_string());

    std::vector<IPAddress> addrs;
    addrs.push_back(addr_res.value());
    handler(addrs);
    return;
  }

  LOG_INFO(get_logger(), "resolving hostname via DNS lookup {}", hostname.c_str());

  s_self =
      shared_from_this(); // safe as there should be only one IOUring instance

  for (auto &req : m_hostname_DNS_requests) {
    if (req.hostname == hostname) {
      LOG_INFO(get_logger(), "duplicate hostname resolve request for {}",
               hostname.c_str());
      req.handlers.push_back(handler);
      return;
    }
  }

  auto &req = m_hostname_DNS_requests.emplace_back(hostname);

  static sigevent sig{};

  req.request->ar_name = hostname.c_str();
  req.request->ar_service = nullptr;
  req.request->ar_request = nullptr;
  req.request->ar_result = nullptr;

  sig.sigev_notify = SIGEV_THREAD;
  sig.sigev_value.sival_ptr = req.request;
  sig.sigev_notify_function = sig_notifier_hostname_resolve;
  sig.sigev_notify_attributes = nullptr;

  getaddrinfo_a(GAI_NOWAIT, req.all_requests, 1, &sig);
}

} // namespace iuring
#include <ifaddrs.h>
#include <thread>

#include "IOUring.hpp"

#include <slogger/ShellUtils.hpp>

#include <iuring/IOUringInterface.hpp>

using namespace std::chrono_literals;

namespace iuring {
void NetworkAdapter::init() {
  retrieve_interface_ip();
  tune();

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    LOG_ERROR(get_logger(), "gethostname() failed");
    abort();
  }
  m_hostname = std::string(hostname);
}

void NetworkAdapter::tune() {
  if (!m_tune) {
    LOG_INFO(get_logger(), "not tuning interface settings");
    return;
  }
  shell::run_cmd("ethtool -C " + m_interface_name + " tx-usecs 1",
                 get_logger(), shell::RunOpt::LOG_ERROR_AS_WARNING);
  shell::run_cmd("ethtool -C " + m_interface_name + " rx-usecs 1",
                 get_logger(), shell::RunOpt::ABORT_ON_ERROR);
  shell::run_cmd("ethtool -C " + m_interface_name + " rx-frames 1",
                 get_logger(), shell::RunOpt::ABORT_ON_ERROR);
  shell::run_cmd("ethtool -C " + m_interface_name + " tx-frames 1",
                 get_logger(), shell::RunOpt::ABORT_ON_ERROR);
}

bool NetworkAdapter::try_get_interface_ip() {
  assert(get_interface_name() != "");

  bool success = false;
  ifaddrs *ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    abort();
  }

  for (ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr)
      continue;

    const auto family = ifa->ifa_addr->sa_family;
    if (std::string(ifa->ifa_name) == "lo") {
      continue;
    }
    if (ifa->ifa_name != get_interface_name()) {
      LOG_INFO(get_logger(), "skip: interface {}", ifa->ifa_name);
      continue;
    }

    switch (family) {
    case AF_INET: {
      IPAddress ip(*(sockaddr_in *)ifa->ifa_addr);
      set_interface_ip4(ip);
      success = true;
      break;
    }

    case AF_INET6: {
      IPAddress ip(*(sockaddr_in6 *)ifa->ifa_addr);
      set_interface_ip6(ip);
      break;
    }

    default:
      LOG_INFO(get_logger(), "FOUND INTERFACE: {} {} ({})", ifa->ifa_name,
               (family == AF_PACKET) ? "AF_PACKET" : (family == AF_INET) ? "AF_INET"
                                                 : (family == AF_INET6)  ? "AF_INET6"
                                                                         : "???",
               family);
      break;
    }
  }

  freeifaddrs(ifaddr);
  return success;
}

void NetworkAdapter::retrieve_interface_ip() {
  // try_get_interface_ip();
  // set_interface_ip4("192.168.1.130");

  while (!get_interface_ip4().has_value()) {
    try_get_interface_ip();
    std::this_thread::sleep_for(1s);
  }
}

std::optional<MacAddress> NetworkAdapter::get_my_mac_address() {
  if (mac_opt) {
    return *mac_opt;
  }

  auto filename = "/sys/class/net/" + get_interface_name() + "/address";
  FILE *f = fopen(filename.c_str(), "r");
  if (!f) {
    LOG_ERROR(get_logger(), "failed to open {}", filename);
    abort();
    return std::nullopt;
  }

  std::array<char, 128> buffer;
  buffer.fill(0);

  int num_bytes_read = 0;
  for (int i = 0; i < 10; i++) {
    num_bytes_read = fread(buffer.data(), 1, buffer.size(), f);
    if (num_bytes_read > 0) {
      break;
    }
  }
  assert(num_bytes_read > 0);
  fclose(f);
  mac_opt = MacAddress{StringUtils::trim(buffer.data())};
  return *mac_opt;
}

} // namespace iuring
#include <cassert>
#include <cerrno>
#include <cstring>
#include <expected>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <slogger/Error.hpp>
#include <slogger/ILogger.hpp>

#include <iuring/IPAddress.hpp>
#include <iuring/MacAddress.hpp>

namespace iuring {
MacAddress::MacAddress(const std::string &mac) {
  std::sscanf(mac.c_str(), "%02hhx:%02hhx:%02hhx:%02hhx:%02hhx:%02hhx",
              &bytes[0], &bytes[1], &bytes[2], &bytes[3], &bytes[4], &bytes[5]);
}

std::expected<IPAddress, error::Error> IPAddress::parse(const std::string &ip_string) {
  sockaddr_in sa4;
  if (inet_pton(AF_INET, ip_string.c_str(), &sa4.sin_addr) == 1) {
    sa4.sin_family = AF_INET;
    return IPAddress(sa4);
  }

  sockaddr_in6 sa6;
  if (inet_pton(AF_INET6, ip_string.c_str(), &sa6.sin6_addr) == 1) {
    sa6.sin6_family = AF_INET6;
    return IPAddress(sa6);
  }

  return std::unexpected(error::Error::HOSTNAME_RESOLVE_FAILED);
}

const std::string MacAddress::to_string(const char sep) const {
  return std::format("{:02x}{}{:02x}{}{:02x}{}{:02x}{}{:02x}{}{:02x}",
                     bytes[0], sep, // layout fix
                     bytes[1], sep, // layout fix
                     bytes[2], sep, // layout fix
                     bytes[3], sep, // layout fix
                     bytes[4], sep, // layout fix
                     bytes[5]);     //
}

std::string IPAddress::to_human_readable_ip_string() const {
  std::array<char, 128> buffer;

  if (auto *v = get_ipv4()) {
    const char *source_name =
        inet_ntop(AF_INET, &v->sin_addr, buffer.data(), buffer.size());
    if (!source_name) {
      source_name = "<INVALID>";
    }
    return std::string(source_name);
  }

  if (auto *v = get_ipv6()) {
    const char *source_name =
        inet_ntop(AF_INET6, &v->sin6_addr, buffer.data(), buffer.size());
    if (!source_name) {
      source_name = "<INVALID>";
    }
    return std::string(source_name);
  }
  return "?.?.?.?";
}

std::string IPAddress::to_human_readable_string() const {
  std::array<char, 128> buffer;

  if (const auto *v = get_ipv4()) {
    const char *source_name =
        inet_ntop(AF_INET, &v->sin_addr, buffer.data(), buffer.size());
    if (!source_name) {
      source_name = "<INVALID>";
    }
    return std::format("v4: {}: port {}", source_name, get_port());
  }

  if (const auto *v = get_ipv6()) {
    const char *source_name =
        inet_ntop(AF_INET6, &v->sin6_addr, buffer.data(), buffer.size());
    if (!source_name) {
      source_name = "<INVALID>";
    }
    return std::format("v6: {}: port {}", source_name, get_port());
  }
  return "?.?.?.?";
}

IPAddress create_sock_addr_in(
    const iuring::IPAddress &addr, const SocketPortID port, logging::ILogger &logger) {
  sockaddr_in dest_addr;
  memset(&dest_addr, 0, sizeof(dest_addr));
  dest_addr.sin_addr =
      iuring::IPAddress::string_to_ipv4_address(addr.to_human_readable_ip_string(), logger);
  dest_addr.sin_family = AF_INET;
  dest_addr.sin_port =
      htons(static_cast<std::underlying_type_t<SocketPortID>>(port));
  return IPAddress(dest_addr);
}

in_addr IPAddress::string_to_ipv4_address(
    const std::string &_ip_address, logging::ILogger &logger) {
  in_addr addr;

  std::string ip_address(_ip_address);
  if (StringUtils::ends_with(ip_address, "/32")) {
    ip_address = ip_address.substr(0, ip_address.length() - 3);
  }

  // fprintf(stderr, "using: {} instead of {}\n", ip_address.c_str(),
  // _ip_address.c_str());
  if (int ret = inet_pton(AF_INET, ip_address.c_str(), &(addr.s_addr));
      ret != 1) {
    if (ret < 0)
      perror("inet_pton - failed");
    else
      LOG_ERROR(logger, "invalid IP address: {}\n", ip_address.c_str());
    abort();
  }
  return addr;
}
} // namespace iuring
#include <cerrno>
#include <cstring>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <slogger/ILogger.hpp>

#include "SocketImpl.hpp"
#include "WorkItem.hpp"
#include "iuring/IOUringInterface.hpp"

namespace iuring {
std::shared_ptr<ISocket> ISocket::create_impl(
    logging::ILogger &logger, const AcceptResult &new_conn) {
  return SocketImpl::create(logger, new_conn);
}

std::shared_ptr<ISocket> ISocket::create_impl(SocketType type,
                                              SocketPortID port, logging::ILogger &logger, SocketKind kind) {
  return SocketImpl::create(type, port, logger, kind);
}

std::shared_ptr<SocketImpl> SocketImpl::create(
    logging::ILogger &logger, const AcceptResult &new_conn) {
  class EnableShared : public SocketImpl {
  public:
    EnableShared(logging::ILogger &logger, const AcceptResult &new_conn)
        : SocketImpl(logger, new_conn) {
    }
  };
  return std::make_shared<EnableShared>(logger, new_conn);
}

std::shared_ptr<SocketImpl> SocketImpl::create(SocketType type,
                                               SocketPortID port, logging::ILogger &logger, SocketKind kind) {
  class EnableShared : public SocketImpl {
  public:
    EnableShared(SocketType type, SocketPortID port,
                 logging::ILogger &logger, SocketKind kind)
        : SocketImpl(type, port, logger, kind) {
    }
  };
  return std::make_shared<EnableShared>(type, port, logger, kind);
}

SocketImpl::SocketImpl(logging::ILogger &logger, const AcceptResult &new_conn)
    : ISocket(iuring::get_type(new_conn), iuring::get_port(new_conn), logger,
              SocketKind::SERVER_STREAM_SOCKET, new_conn.m_new_fd) {
  memset(&m_mreq, 0, sizeof(m_mreq));
  assert(get_fd() > 0);
}

namespace {
int create_socket(logging::ILogger &logger, SocketType type) {
  int non_blocking_option = 0;
  if (false) {
    non_blocking_option |= SOCK_NONBLOCK;
  }

  int fd = -1;

  switch (type) {
  case SocketType::UNKNOWN:
    assert(false);
    break;

  case SocketType::IPV4_UDP:
    fd = socket(AF_INET, SOCK_DGRAM | non_blocking_option, 0);
    LOG_DEBUG(logger, "socket-v4 {} with dgram type!", fd);
    break;
  case SocketType::IPV4_TCP:
    fd = socket(AF_INET, SOCK_STREAM | non_blocking_option, 0);
    LOG_DEBUG(logger, "socket-v4 {} with stream type!", fd);
    break;
  case SocketType::IPV6_UDP:
    fd = socket(AF_INET6, SOCK_DGRAM | non_blocking_option, 0);
    LOG_DEBUG(logger, "socket-v6 {} with dgram type!", fd);
    abort();
    break;
  case SocketType::IPV6_TCP:
    fd = socket(AF_INET6, SOCK_STREAM | non_blocking_option, 0);
    LOG_DEBUG(logger, "socket-v6 {} with stream type!", fd);
    break;
  }

  assert(fd >= 0);
  return fd;
}
} // namespace

SocketImpl::SocketImpl(SocketType type, SocketPortID port,
                       logging::ILogger &logger, SocketKind kind)
    : ISocket(type, port, logger, kind, create_socket(logger, type)) {
  memset(&m_mreq, 0, sizeof(m_mreq));

  int set_option_on = 1;
  // it is important to do "reuse address" before bind, not after
  [[maybe_unused]] int res = setsockopt(get_fd(), SOL_SOCKET, SO_REUSEADDR,
                                        (char *)&set_option_on, sizeof(set_option_on));
  assert(res == 0);

  switch (kind) {
  case SocketKind::UNICAST_CLIENT_SOCKET: {
    // local_bind(static_cast<SocketPortID>(9090));

    int val = 1;
    int ret =
        setsockopt(get_fd(), SOL_SOCKET, SO_REUSEPORT, &val, sizeof(val));
    if (ret == -1) {
      perror("setsockopt()");
      abort();
    }
    break;
  }

  case SocketKind::MULTICAST_PACKET_SOCKET: {
    local_bind(port);

    int ttl = 1;
    if (setsockopt(
            get_fd(), IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl))) {
      perror("setsockopt IP_MULTICAST_TTL failed");
      abort();
    }
    break;
  }

  case SocketKind::SERVER_STREAM_SOCKET: {
    const auto tmp_port =
        static_cast<std::underlying_type_t<SocketPortID>>(port);

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(tmp_port);

    int err = bind(
        get_fd(), (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (err < 0) {
      LOG_ERROR(get_logger(), "bind error: {} (port {})", strerror(errno),
                tmp_port);
      abort();
    }

    err = ::listen(get_fd(), 1024);
    if (err < 0) {
      LOG_INFO(get_logger(), "listen error: {}, port {}", strerror(errno),
               tmp_port);
      abort();
    }
    break;
  }
  }
}

void SocketImpl::dump_info() {
  assert(get_fd() >= 0);
  sockaddr_storage s;
  socklen_t sz = sizeof(s);

  if (getsockname(get_fd(), (struct sockaddr *)&s, &sz)) {
    LOG_ERROR(get_logger(), "getsockname failed\n");
    return;
  }

  const auto *sa = (struct sockaddr_in *)&s;

  const auto port = ntohs(sa->sin_port);
  const auto addr = sa->sin_addr;
  LOG_DEBUG(get_logger(), "DOUBLE CHECK -----> port bound to {}: {}", port,
            inet_ntoa(addr));
}

void SocketImpl::send(const std::shared_ptr<iuring::IOUringInterface> &io,
                      const std::string &reply_msg, const iuring::send_callback_func_t &cb) {
  auto wi = io->ackuire_send_workitem(this->shared_from_this());
  auto &pkt = wi->get_send_packet();
  pkt.append(reply_msg);
  wi->submit_stream_data(cb);
}

int SocketImpl::mcast_bind() {
  assert(get_fd() >= 0);

  ip_mreq req = m_mreq;

  int err =
      setsockopt(get_fd(), IPPROTO_IP, IP_MULTICAST_IF, &req, sizeof(req));
  if (err) {
    perror("setsockopt IP_MULTICAST_IF failed: %m");
    return -1;
  }
  return 0;
}

void SocketImpl::join_multicast_group(
    const iuring::IPAddress &ip_address,
    const iuring::IPAddress &source_iface) {
  assert(get_fd() >= 0);

  LOG_DEBUG(get_logger(), "PTP: join_multicast_group: '{}', interface '{}'\n",
            ip_address, source_iface);

  m_mreq.imr_multiaddr =
      IPAddress::string_to_ipv4_address(ip_address.to_human_readable_ip_string(), get_logger());
  m_mreq.imr_interface =
      IPAddress::string_to_ipv4_address(source_iface.to_human_readable_ip_string(), get_logger());

  if (int ret = setsockopt(
          get_fd(), IPPROTO_IP, IP_ADD_MEMBERSHIP, &m_mreq, sizeof(m_mreq));
      ret < 0) {
    perror("setsockopt");
    abort();
  }

  int off = 0;
  if (int err = setsockopt(
          get_fd(), IPPROTO_IP, IP_MULTICAST_LOOP, &off, sizeof(off));
      err) {
    perror("setsockopt IP_MULTICAST_LOOP failed");
    abort();
  }
}

void SocketImpl::leave_multicast_group() {
  assert(get_fd() >= 0);
  LOG_DEBUG(get_logger(), "PTP: leave_multicast_group");
  if (int ret = setsockopt(
          get_fd(), IPPROTO_IP, IP_DROP_MEMBERSHIP, &m_mreq, sizeof(m_mreq));
      ret < 0) {
    perror("setsockopt");
    exit(1);
  }
}

void SocketImpl::local_bind(SocketPortID port_id) {
  assert(get_fd() >= 0);

  LOG_DEBUG(get_logger(), "PTP: binding interface to port {}\n", port_id);
  sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(
      static_cast<std::underlying_type_t<iuring::SocketPortID>>(port_id));

  if (int ret = ::bind(get_fd(), (sockaddr *)&addr, sizeof(addr)); ret < 0) {
    perror("bind failed for local_bind");
    LOG_ERROR(get_logger(), "failed to bind to port {}, exiting (fd={})",
              port_id, get_fd());
    if (port_id < SocketPortID::LAST_PRIVILEDGED_PORT_ID) {
      LOG_ERROR(get_logger(), "are you using 'sudo'?");
    }
    abort();
  }

  dump_info();
}
} // namespace iuring
#include <iuring/IOUringInterface.hpp>

namespace iuring {
const char *WorkItem::type_to_string(Type t) {
  switch (t) {
  case Type::ACCEPT:
    return "accept";
  case Type::CONNECT:
    return "connect";
  case Type::RECV:
    return "recv";
  case Type::SEND_STREAM_DATA:
    return "send_stream_data";
  case Type::SEND_WORKPACKET:
    return "send_workpacket";
  case Type::UNKNOWN:
    return "unknown";
  case Type::CLOSE:
    return "close";
  }
  return "<unknown type of work item>";
}

void WorkItem::submit(const recv_callback_func_t &cb) {
  m_callback = cb;
  m_work_type = Type::RECV;
  m_io_ring->submit(*this);
}

void WorkItem::submit_packet(
    const DatagramSendParameters &params, const send_callback_func_t &cb) {
  m_params = params;
  m_callback = cb;
  m_work_type = Type::SEND_WORKPACKET;
  m_io_ring->submit(*this);
}

void WorkItem::submit_stream_data(const send_callback_func_t &cb) {
  m_callback = cb;
  m_work_type = Type::SEND_STREAM_DATA;
  m_io_ring->submit(*this);
}

void WorkItem::submit(const accept_callback_func_t &cb) {
  m_callback = cb;
  m_work_type = Type::ACCEPT;
  m_io_ring->submit(*this);
}

void WorkItem::submit(
    const IPAddress &target, const connect_callback_func_t &cb) {
  LOG_INFO(
      get_logger(), "connecting to {}", target.to_human_readable_ip_string());

  m_connect_sock_len = target.size_sockaddr();
  assert(sizeof(m_buffer_for_uring) >= m_connect_sock_len);
  memcpy(&m_buffer_for_uring, target.data_sockaddr(), m_connect_sock_len);

  m_callback = cb;
  m_work_type = Type::CONNECT;
  m_io_ring->submit(*this);
}

void WorkItem::submit(const close_callback_func_t &cb) {
  m_callback = cb;
  m_work_type = Type::CLOSE;
  m_io_ring->submit(*this);
}

SocketType get_type(const AcceptResult &res) {
  if (res.m_address.get_ipv4()) {
    return SocketType::IPV4_TCP;
  }
  if (res.m_address.get_ipv6()) {
    return SocketType::IPV6_TCP;
  }

  return SocketType::UNKNOWN;
}

SocketPortID get_port(const AcceptResult &res) {
  if (auto addr = res.m_address.get_ipv4()) {
    return static_cast<SocketPortID>(addr->sin_port);
  }
  if (auto addr = res.m_address.get_ipv6()) {
    return static_cast<SocketPortID>(addr->sin6_port);
  }
  return SocketPortID::UNKNOWN;
}

} // namespace iuring

#include "WorkPool.hpp"
#include <iuring/IOUringInterface.hpp>

namespace iuring {
std::shared_ptr<WorkItem> WorkPool::get_work_item(work_item_id_t id) {
  std::lock_guard lock(m_mutex);
  // assert(id >= 0);
  assert(id <= m_work_items.size());
  auto work_item = m_work_items[id];
  if (!work_item) {
    return nullptr;
  }

  assert(work_item != nullptr);
  assert(!work_item->is_free());
  return work_item;
}

void WorkPool::free_work_item(work_item_id_t id) {
  std::lock_guard lock(m_mutex);
  assert(id <= m_work_items.size());
  auto work_item = m_work_items[id];
  assert(work_item != nullptr);

  assert(!work_item->is_free());
  work_item->mark_is_free();

  m_work_items[id] = nullptr;

  m_free_ids.push(id);
}

std::shared_ptr<WorkItem> WorkPool::internal_alloc_work_item(
    const std::shared_ptr<ISocket> &socket,
    const std::shared_ptr<iuring::IOUringInterface> &network,
    const char *descr) {
  if (m_free_ids.empty()) {
    const auto id = m_work_items.size();
    if (id > 64) {
      LOG_ERROR_ONCE(get_logger(), "potential FD leak!!!");
    }
    LOG_DEBUG(get_logger(), "  NEW: id = {} ({})", id, descr);

    auto ret = std::make_shared<WorkItem>(get_logger(), network, id, descr, socket);
    assert(!ret->is_free());
    m_work_items.push_back(ret);
    return ret;
  } else {
    work_item_id_t id = m_free_ids.top();
    m_free_ids.pop();
    LOG_DEBUG(get_logger(),
              "allocating work item from prepped-queue: {} ({})", id, descr);

    auto ret = std::make_shared<WorkItem>(get_logger(), network, id, descr, socket);
    assert(!ret->is_free());
    m_work_items[id] = ret;
    return ret;
  }
}

std::shared_ptr<WorkItem> WorkPool::alloc_send_work_item(
    const std::shared_ptr<ISocket> &socket,
    const std::shared_ptr<iuring::IOUringInterface> &network,
    const char *descr) {
  std::lock_guard lock(m_mutex);
  auto wi = internal_alloc_work_item(socket, network, descr);
  assert(wi);
  return wi;
}

std::shared_ptr<WorkItem> WorkPool::alloc_recv_work_item(
    const std::shared_ptr<ISocket> &socket,
    const std::shared_ptr<iuring::IOUringInterface> &network,
    const recv_callback_func_t &callback, const char *descr) {
  std::lock_guard lock(m_mutex);
  auto wi = internal_alloc_work_item(socket, network, descr);
  assert(wi);
  wi->submit(callback);
  return wi;
}

std::shared_ptr<WorkItem> WorkPool::alloc_accept_work_item(
    const std::shared_ptr<ISocket> &socket,
    const std::shared_ptr<iuring::IOUringInterface> &network,
    const accept_callback_func_t &callback, const char *descr) {
  std::lock_guard lock(m_mutex);
  auto wi = internal_alloc_work_item(socket, network, descr);
  assert(wi);
  wi->submit(callback);
  return wi;
}

std::shared_ptr<WorkItem> WorkPool::alloc_connect_work_item(
    const IPAddress &target,
    const std::shared_ptr<ISocket> &socket,
    const std::shared_ptr<iuring::IOUringInterface> &network,
    const connect_callback_func_t &callback, const char *descr) {
  std::lock_guard lock(m_mutex);
  auto wi = internal_alloc_work_item(socket, network, descr);
  assert(wi);
  wi->submit(target, callback);
  return wi;
}

std::shared_ptr<WorkItem> WorkPool::alloc_close_work_item(
    const std::shared_ptr<ISocket> &socket,
    const std::shared_ptr<iuring::IOUringInterface> &network,
    const close_callback_func_t &callback, const char *descr) {
  std::lock_guard lock(m_mutex);
  auto wi = internal_alloc_work_item(socket, network, descr);
  assert(wi);
  wi->submit(callback);
  return wi;
}

} // namespace iuring

#include <expected>
#include <functional>
#include <string>

/**
 * @file CompletionCallbacks.hpp
 * @brief Defines callback function types for network operations.
 */

namespace iuring {
enum class [[nodiscard]] ReceivePostAction { NONE,
                                             RE_SUBMIT };

struct AcceptResult {
  int m_new_fd;
  IPAddress m_address;
};

class SendResult {
public:
  explicit SendResult(int s) : m_status(s) {}

  std::expected<int, error::Error> to_expected() const {
    if (m_status >= 0) {
      return std::expected<int, error::Error>(m_status);
    } else {
      return std::unexpected<error::Error>(
          error::errno_to_error(-m_status));
    }
  }

private:
  int m_status;
};

class ConnectResult {
public:
  ConnectResult(int status, const IPAddress &addr)
      : m_status(status), m_address(addr) {
  }

  std::expected<IPAddress, error::Error> to_expected() const {
    if (m_status >= 0) {
      return m_address;
    } else {
      return std::unexpected<error::Error>(
          error::errno_to_error(-m_status));
    }
  }

private:
  int m_status;
  IPAddress m_address;
};

class CloseResult {
public:
  explicit CloseResult(int s) : m_status(s) {}

  std::expected<int, error::Error> to_expected() const {
    if (m_status >= 0) {
      return std::expected<int, error::Error>(m_status);
    } else {
      return std::unexpected<error::Error>(
          error::errno_to_error(-m_status));
    }
  }

private:
  int m_status;
};

class ReceivedMessage;

using recv_callback_func_t =
    std::function<ReceivePostAction(const ReceivedMessage &msg)>;

using send_callback_func_t = std::function<void(const SendResult &)>;

using accept_callback_func_t =
    std::function<void(const AcceptResult &new_conn)>;

using connect_callback_func_t =
    std::function<void(const ConnectResult &result)>;

using close_callback_func_t = std::function<void(const CloseResult &result)>;

} // namespace iuring

/**
 * @file IOUringInterface.hpp
 * @brief Defines the IOUringInterface for asynchronous I/O operations.
 */

#include <expected>
#include <functional>
#include <memory>
#include <vector>

#include <slogger/Error.hpp>
#include <slogger/ILogger.hpp>

#include "CompletionCallbacks.hpp"
#include "IPAddress.hpp"
#include "ISocket.hpp"
#include "IWorkItem.hpp"
#include "MacAddress.hpp"
#include "NetworkAdapter.hpp"

namespace iuring {
class IOUringInterface {
public:
  virtual ~IOUringInterface() {}

  using resolve_hostname_arg_t = std::expected<std::vector<IPAddress>, error::Error>;
  using resolve_hostname_callback_func_t = std::function<void(
      const resolve_hostname_arg_t &result)>;

  static std::shared_ptr<IOUringInterface> create_impl(logging::ILogger &logger, NetworkAdapter &adapter);

  virtual error::Error init() = 0;

  virtual error::Error poll_completion_queues() = 0;

  virtual void resolve_hostname(const std::string &hostname,
                                const resolve_hostname_callback_func_t &handler) = 0;

  virtual void submit_connect(const std::shared_ptr<ISocket> &socket,
                              const IPAddress &target, connect_callback_func_t handler) = 0;

  /** This accepts new connections from other machines.
   * Note that this requires that the socket is opened with
   *  SocketKind::SERVER_STREAM_SOCKET
   * As only server sockets can accept new connections.
   * We check this by asserting the correct behavior here to safeguard this.
   */
  virtual void submit_accept(const std::shared_ptr<ISocket> &socket,
                             accept_callback_func_t handler) = 0;

  virtual void submit_recv(const std::shared_ptr<ISocket> &socket,
                           recv_callback_func_t handler) = 0;

  /** The steps for sending a packet:
   *      - This returns a work-item where you can retrieve the SendPacket
   * object from
   *      - Then with that send packet you append your dara
   *      - Then you call submit on the work-item.
   *      - The WorkItem::submit() method then has the callback arg.
   */
  virtual std::shared_ptr<IWorkItem> ackuire_send_workitem(
      const std::shared_ptr<ISocket> &socket) = 0;

  /** used to submit the submit_sent returned
   * work item.
   */
  virtual void submit(IWorkItem &item) = 0;

  virtual void submit_close(const std::shared_ptr<ISocket> &socket,
                            close_callback_func_t handler) = 0;
};

} // namespace iuring

/**
 * @file IPAddress.hpp
 * @brief Defines the IPAddress class for handling IPv4 and IPv6 addresses.
 */

#include <cassert>
#include <cstring>
#include <net/if.h>
#include <netinet/in.h>
#include <optional>
#include <poll.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include <expected>
#include <string>

#include <slogger/Error.hpp>
#include <slogger/ILogger.hpp>
#include <slogger/StringUtils.hpp>

#include <iuring/NetworkProtocols.hpp>

namespace iuring {
class IPAddress {
public:
  IPAddress() = default;

  explicit IPAddress(const in_addr &sa, SocketPortID port) {
    sockaddr_in sa_in;
    memset(&sa_in, 0, sizeof(sa));

    sa_in.sin_family = AF_INET;
    sa_in.sin_port = htons(static_cast<uint16_t>(port));
    sa_in.sin_addr = sa;

    m_in4 = sa_in;
  }

  explicit IPAddress(const in6_addr &sa, SocketPortID port) {
    sockaddr_in6 sa_in;
    memset(&sa_in, 0, sizeof(sa));

    sa_in.sin6_family = AF_INET6;
    sa_in.sin6_port = htons(static_cast<uint16_t>(port));
    sa_in.sin6_addr = sa;

    m_in6 = sa_in;
  }

  explicit IPAddress(const sockaddr_in &sa)
      : m_in4(sa) {
  }

  explicit IPAddress(const sockaddr_in6 &sa)
      : m_in6(sa) {
  }

  IPAddress(const sockaddr_storage &sa, socklen_t len)
      : m_in4(len == sizeof(sockaddr_in) ? std::nullopt : std::optional<sockaddr_in>(*(sockaddr_in *)&sa)), m_in6(len == sizeof(sockaddr_in6) ? std::nullopt : std::optional<sockaddr_in6>(*(sockaddr_in6 *)&sa)) {
    assert((len == sizeof(sockaddr_in)) || (len == sizeof(sockaddr_in6)));
  }

  void set_port(SocketPortID port) {
    if (auto *a = get_mut_ipv4()) {
      a->sin_port = htons(static_cast<uint16_t>(port));
      return;
    }

    if (auto *a = get_mut_ipv6()) {
      a->sin6_port = htons(static_cast<uint16_t>(port));
      return;
    }
    abort();
  }

  SocketPortID get_port() const {
    if (auto *a = get_ipv4()) {
      const auto p = htons(static_cast<uint16_t>(a->sin_port));
      return static_cast<SocketPortID>(p);
    }

    if (auto *a = get_ipv6()) {
      const auto p = htons(static_cast<uint16_t>(a->sin6_port));
      return static_cast<SocketPortID>(p);
    }
    abort();
  }

  bool valid() const {
    if (get_ipv4())
      return true;
    if (get_ipv6())
      return true;
    return false;
  }

  /** IP address and port */
  std::string to_human_readable_string() const;

  /** just the IP address is returned */
  std::string to_human_readable_ip_string() const;

  std::string to_string() const {
    return to_human_readable_ip_string();
  }

  const void *data_sockaddr() const {
    if (const auto *a = get_ipv4())
      return a;
    if (const auto *b = get_ipv6())
      return b;
    abort();
  }

  socklen_t size_sockaddr() const {
    if (const auto *a = get_ipv4())
      return sizeof(*a);
    if (const auto *b = get_ipv6())
      return sizeof(*b);
    abort();
  }

  const void *data_addr() const {
    if (const auto *a = get_ipv4())
      return &a->sin_addr.s_addr;
    if (const auto *b = get_ipv6())
      return &b->sin6_addr;
    abort();
  }

  size_t size_addr() const {
    if (const auto *a = get_ipv4())
      return sizeof(a->sin_addr);
    if (const auto *b = get_ipv6())
      return sizeof(b->sin6_addr);
    abort();
  }

  /** returns null if not ipv4 */
  const sockaddr_in *get_ipv4() const {
    if (m_in4) {
      return &*m_in4;
    }

    return nullptr;
  }

  /** returns null if not ipv6 */
  const sockaddr_in6 *get_ipv6() const {
    if (m_in6) {
      return &*m_in6;
    }
    return nullptr;
  }

  sockaddr_in *get_mut_ipv4() {
    if (m_in4) {
      return &*m_in4;
    }

    return nullptr;
  }

  sockaddr_in6 *get_mut_ipv6() {
    if (m_in6) {
      return &*m_in6;
    }
    return nullptr;
  }

  uint64_t get_hash() const {
    if (m_in4) {
      return *(uint32_t *)&m_in4->sin_addr;
    } else if (m_in6) {
      const uint32_t *a = m_in6->sin6_addr.__in6_u.__u6_addr32;

      const uint64_t ret = (((uint64_t)a[0]) << 0) |
                           (((uint64_t)a[1]) << 32) | (((uint64_t)a[2]) << 0) |
                           (((uint64_t)a[3]) << 32);
      return ret;
    }

    abort();
  }

  bool operator==(const IPAddress &other) const {
    if (m_in4.has_value() and other.m_in4.has_value()) {
      const auto &v1 = m_in4.value();
      const auto &v2 = other.m_in4.value();
      return v1.sin_port == v2.sin_port &&
             memcmp(&v1.sin_addr, &v2.sin_addr, sizeof(v1.sin_addr)) == 0 &&
             v1.sin_family == v2.sin_family;
    }
    if (m_in6.has_value() and other.m_in6.has_value()) {
      const auto &v1 = m_in6.value();
      const auto &v2 = other.m_in6.value();
      return v1.sin6_port == v2.sin6_port &&
             memcmp(&v1.sin6_addr, &v2.sin6_addr, sizeof(v1.sin6_addr)) == 0 &&
             v1.sin6_family == v2.sin6_family;
    }
    return false;
  }

  bool operator<(const IPAddress &addr) const {
    return get_hash() < addr.get_hash();
  }

  static std::expected<IPAddress, error::Error> parse(const std::string &ip_string);

public:
  static in_addr string_to_ipv4_address(
      const std::string &_ip_address, logging::ILogger &logger);

  std::optional<uint8_t> get_subnet_mask() const {
    return m_subnet_mask;
  }

  void set_subnet_mask(uint8_t mask) {
    m_subnet_mask = mask;
  }

private:
  std::optional<sockaddr_in> m_in4;
  std::optional<uint8_t> m_subnet_mask;
  std::optional<sockaddr_in6> m_in6;
};

/** util func for converting a 'a.b.c.d' IP address and
 * port to an IPAddress object
 */
IPAddress create_sock_addr_in(
    const iuring::IPAddress &addr, const SocketPortID port, logging::ILogger &logger);

} // namespace iuring

template <>
struct std::formatter<iuring::IPAddress> {
  constexpr auto parse(std::format_parse_context &ctx) {
    return ctx.begin();
  }

  auto format(const iuring::IPAddress &c, std::format_context &ctx) const {
    return std::format_to(ctx.out(), "{}", c.to_human_readable_string());
  }
};
#pragma once

#include <memory>

#include <iuring/ISocket.hpp>

namespace iuring {

/** A factory class for creating ISocket instances.
 *
 * - In the unit tests this is overriden with an instance that returns mocks.
 * - normally, this is overriden by iuring::SocketFactory.
 */
class ISocketFactory {
public:
  virtual std::shared_ptr<ISocket> create_impl(SocketType type,
                                               SocketPortID port, logging::ILogger &logger, SocketKind kind) = 0;

  virtual std::shared_ptr<ISocket> create_impl(
      logging::ILogger &logger, const AcceptResult &res) = 0;
};
} // namespace iuring

/**
 * @file ISocket.hpp
 * @brief Defines the ISocket interface for network sockets.
 *
 * This interface provides the basic functionalities for different types of
 * network sockets, including multicast binding and joining multicast groups.
 * The real implementations will derive from this interface.
 * The main implementation is in SocketImpl.hpp
 */

#include <net/if.h>
#include <netinet/in.h>
#include <optional>
#include <poll.h>
#include <queue>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cstring>

#include <slogger/ILogger.hpp>
#include <slogger/StringUtils.hpp>

#include <iuring/CompletionCallbacks.hpp>
#include <iuring/IPAddress.hpp>

namespace iuring {
class AcceptResult;
class IOUringInterface;

/** use IConnectionData to add a context to the use of the ISocket.
 * For example, when using SSL, attach the encrypted buffers to the ISocket
 * or attach the socket's state.
 */
class IConnectionData {
};

class ISocket : public std::enable_shared_from_this<ISocket> {
public:
  ISocket(SocketType type, SocketPortID port, logging::ILogger &logger,
          SocketKind kind, int fd)
      : m_type(type), m_port(port), m_logger(logger), m_kind(kind), m_fd(fd) {
  }

  /** @brief Send the msg to 'io' and call 'cb' once done.
   */
  virtual void send(const std::shared_ptr<iuring::IOUringInterface> &io,
                    const std::string &msg, const send_callback_func_t &cb) = 0;

  virtual int mcast_bind() = 0;

  virtual void join_multicast_group(
      const iuring::IPAddress &ip_address, const iuring::IPAddress &source_iface) = 0;

  virtual ~ISocket() = default;

  int get_fd() const {
    return m_fd;
  }

  SocketPortID get_port() const {
    return m_port;
  }

  SocketKind get_kind() const {
    return m_kind;
  }

  bool is_stream() const {
    switch (m_type) {
    case SocketType::IPV4_TCP:
    case SocketType::IPV6_TCP:
      return true;
    case SocketType::UNKNOWN:
    case SocketType::IPV4_UDP:
    case SocketType::IPV6_UDP:
      return false;
    }
    return false;
  }

  SocketType get_type() const {
    return m_type;
  }

  logging::ILogger &get_logger() {
    return m_logger;
  }

  void set_connection_data(const std::shared_ptr<IConnectionData> &connection_data) {
    m_connection_data = connection_data;
  }

  const std::shared_ptr<IConnectionData> &get_connection_data() {
    return m_connection_data;
  }

private:
  SocketType m_type;
  SocketPortID m_port;
  logging::ILogger &m_logger;
  SocketKind m_kind;
  int m_fd;

  std::shared_ptr<IConnectionData> m_connection_data;

private:
  friend class SocketFactoryImpl;

  static std::shared_ptr<ISocket> create_impl(SocketType type,
                                              SocketPortID port, logging::ILogger &logger, SocketKind kind);

  static std::shared_ptr<ISocket> create_impl(
      logging::ILogger &logger, const AcceptResult &res);
};

} // namespace iuring

/**
 * @file IWorkItem.hpp
 * @brief Defines the IWorkItem interface for work items in asynchronous I/O
 * operations.
 *
 * This interface provides the basic functionalities for different types of
 * work items, such as sending, receiving, connecting, etc.
 */

#include <functional>

#include "CompletionCallbacks.hpp"
#include "NetworkProtocols.hpp"
#include "ReceivedMessage.hpp"
#include "SendPacket.hpp"

namespace iuring {
struct DatagramSendParameters {
  IPAddress destination_address;
  dscp_t dscp;
  timetolive_t ttl;
};

class IWorkItem {
public:
  enum class Type {
    UNKNOWN,
    ACCEPT,
    SEND_STREAM_DATA,
    SEND_WORKPACKET,
    RECV,
    CONNECT,
    CLOSE
  };

  virtual ~IWorkItem() {}

  Type get_type() const {
    return m_work_type;
  }

  virtual SendPacket &get_send_packet() = 0;

  /** @brief Submits the work item for processing.
   */
  virtual void submit_stream_data(const send_callback_func_t &cb) = 0;

  /** @brief Submits the work item for processing.
   */
  virtual void submit_packet(const DatagramSendParameters &params,
                             const send_callback_func_t &cb) = 0;

  /** @brief Get the socket associated with this work item.
   *
   * Typically not needed. Can be used if you have multiple sockets and want to
   * know which one this work item is for.
   */
  virtual std::shared_ptr<ISocket> get_socket() const = 0;

protected:
  Type m_work_type = Type::UNKNOWN;
};
} // namespace iuring
#pragma once

#include <array>
#include <string>

#include <cstdlib>

namespace iuring {
class MacAddress {
public:
  explicit MacAddress(const std::string &mac);

  explicit MacAddress(
      uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4, uint8_t b5) {
    bytes[0] = b0;
    bytes[1] = b1;
    bytes[2] = b2;
    bytes[3] = b3;
    bytes[4] = b4;
    bytes[5] = b5;
  }

  const std::string to_string(const char sep = ':') const;

  const std::array<uint8_t, 6> &to_bytes() const {
    return bytes;
  }

private:
  std::array<uint8_t, 6> bytes{};
};
} // namespace iuring

#include <optional>
#include <string>

#include <slogger/ILogger.hpp>

#include <iuring/IPAddress.hpp>

#include "MacAddress.hpp"

namespace iuring {
class NetworkAdapter {
public:
  NetworkAdapter(
      logging::ILogger &logger, const std::string &interface_name, bool tune)
      : m_logger(logger), m_interface_name(interface_name), m_tune(tune) {
  }

  void init();

  void tune();

  void set_interface_name(const std::string &interface_name) {
    m_interface_name = interface_name;
    LOG_INFO(get_logger(), "interface name set to {}", interface_name);
  }

  const std::string &get_hostname() const {
    return m_hostname;
  }

  void set_interface_ip4(const iuring::IPAddress &ip) {
    m_interface_ip4 = ip;
    LOG_INFO(get_logger(), "interface IP4 set to {}", ip);
  }

  void set_interface_ip6(const iuring::IPAddress &ip) {
    m_interface_ip6 = ip;
    LOG_INFO(get_logger(), "interface IP6 set to {}", ip);
  }

  /**
   * returns eth line eth0, eth1, etc. or wlan0
   */
  const std::string &get_interface_name() const {
    return m_interface_name;
  }

  std::optional<MacAddress> get_my_mac_address();

  /** @return the IP address we're currently bound to on our selected
   * interface (eth0 in '--dev eth0')
   */
  const std::optional<iuring::IPAddress> get_interface_ip4() const {
    return m_interface_ip4;
  }

  /** @return the IP address we're currently bound to on our selected
   * interface (eth0 in '--dev eth0')
   */
  const std::optional<iuring::IPAddress> get_interface_ip6() const {
    return m_interface_ip6;
  }

private:
  std::string m_hostname;
  logging::ILogger &m_logger;

  std::optional<iuring::IPAddress> m_interface_ip4;
  std::optional<iuring::IPAddress> m_interface_ip6;

  // eth0
  std::string m_interface_name;
  bool m_tune = true;
  std::optional<MacAddress> mac_opt;

  bool try_get_interface_ip();
  void retrieve_interface_ip();

  logging::ILogger &get_logger() {
    return m_logger;
  }
};
} // namespace iuring

#include <cassert>
#include <chrono>
#include <cstdint>
#include <stdlib.h>

#include "SocketPortID.hpp"

namespace iuring {

enum class SocketKind {
  MULTICAST_PACKET_SOCKET,
  SERVER_STREAM_SOCKET,
  UNICAST_CLIENT_SOCKET
};

enum class SocketType {
  UNKNOWN,

  IPV4_UDP,
  IPV4_TCP,

  IPV6_UDP,
  IPV6_TCP
};

enum class timetolive_t : uint8_t {
  PTP_TTL = 16,
  RTP_TTL = 32,
  NORMAL_TTL = 58,
  MDNS_TTL = 255
};

enum class dscp_t : uint8_t {
  CS0 = 0,
  CS1 = 8,
  CS2 = 16,
  CS3 = 24,
  CS4 = 32,
  CS5 = 40,
  CS6 = 48,
  CS7 = 56,

  AF11 = 10,
  AF12 = 12,
  AF13 = 14,

  AF21 = 18,
  AF22 = 20,
  AF23 = 22,

  AF31 = 26,
  AF32 = 28,
  AF33 = 30,

  AF41 = 34,
  AF42 = 36,
  AF43 = 38,

  VOICE_ADMIT = 44,
  EXPEDITED_FORWARDING = 46,

  BEST_EFFORT = CS0,

  // RAVENNA and Dante use other DSCP defaults (CS6 (48) for PTP, EF (46) for
  // audio),

  AES67_PTP_EVENT = EXPEDITED_FORWARDING,
  AES67_PTP_GENERAL = BEST_EFFORT,
  AES67_RTP = AF41,

  RAV_DANTE_PTP_EVENT = CS6,
  RAV_DANTE_PTP_GENERAL = BEST_EFFORT,
  RAV_DANTE_RTP = EXPEDITED_FORWARDING
};
} // namespace iuring

template <>
struct std::formatter<iuring::SocketPortID> {
  constexpr auto parse(std::format_parse_context &ctx) {
    return ctx.begin();
  }

  auto format(iuring::SocketPortID c, std::format_context &ctx) const {
    return std::format_to(ctx.out(), "{}", static_cast<uint16_t>(c));
  }
};

#pragma once

#include <cstdint>
#include <string>

#include <iuring/IPAddress.hpp>

namespace iuring {
class ReceivedMessage {
public:
  ReceivedMessage(const uint8_t *data, size_t size, const IPAddress &sa)
      : m_data(data), m_size(size), m_source_address(sa) {
  }

  std::string to_string() const {
    return std::string((const char *)begin(), get_size());
  }

  const uint8_t *begin() const {
    return m_data;
  }

  bool is_empty() const {
    return get_size() == 0;
  }

  size_t get_size() const {
    return m_size;
  }

  const uint8_t *end() const {
    return m_data + m_size;
  }

  const IPAddress &get_source_address() const {
    return m_source_address;
  }

private:
  const uint8_t *m_data;
  size_t m_size;
  IPAddress m_source_address;
};

} // namespace iuring

#include <arpa/inet.h>
#include <stdlib.h>

#include <array>
#include <cassert>
#include <string>

namespace iuring {
class SendPacket {
public:
  void append_byte(uint8_t b) {
    append(&b, 1);
  }

  void append_uint16(uint16_t v) {
    const uint16_t n = ntohs(v);
    append((const uint8_t *)&n, sizeof(n));
  }

  void append_uint32(uint32_t v) {
    const uint32_t n = ntohl(v);
    append((const uint8_t *)&n, sizeof(n));
  }

  template <typename T>
  void append(const T &data) {
    append((const uint8_t *)&data, sizeof(data));
  }

  void append(const std::string &data) {
    append((const uint8_t *)data.c_str(), data.length());
  }

  void append(const char *data) {
    append((const uint8_t *)data, strlen(data));
  }

  void append(const uint8_t *data, size_t len) {
    assert((m_size + len) < m_buf.size());
    memcpy(&m_buf[m_size], data, len);
    m_size += len;
  }

  template <class T, class... Args>
  void emplace_back(Args &&...args) {
    assert((m_size + sizeof(T)) < m_buf.size());
    auto *ptr = &m_buf[m_size];

    memset(ptr, 0, sizeof(T)); // NOTE: memset possibly superfluous if T's ctor is ok
    new (ptr) T(args...);

    m_size += sizeof(T);
  }

  void reset() {
    memset(m_buf.data(), 0, m_size);
    m_size = 0;
  }

  void clean_proper() {
    m_size = 0;
    m_buf.fill(0);
  }

  size_t size() const {
    return m_size;
  }

  const uint8_t *data() const {
    return m_buf.data();
  }

  std::string to_string() const {
    return std::string(reinterpret_cast<const char *>(m_buf.data()), m_size);
  }

private:
  size_t m_size = 0;
  std::array<uint8_t, 4096> m_buf;
};

} // namespace iuring

#include <memory>

#include <iuring/ISocketFactory.hpp>

namespace iuring {

/** creates real impls */
class SocketFactoryImpl : public ISocketFactory {
public:
  std::shared_ptr<ISocket> create_impl(SocketType type, SocketPortID port,
                                       logging::ILogger &logger, SocketKind kind) override {
    return ISocket::create_impl(type, port, logger, kind);
  }

  std::shared_ptr<ISocket> create_impl(
      logging::ILogger &logger, const AcceptResult &res) override {
    return ISocket::create_impl(logger, res);
  }
};
} // namespace iuring

namespace iuring {
enum class SocketPortID : u_int16_t {
  SSH_PORT = 22,

  UNENCRYPTED_WEB_PORT = 80,

  PTP_PORT_EVENT = 319,
  PTP_PORT_GENERAL = 320,

  ENCRYPTED_WEB_PORT = 443,

  // real-time streaming protocol
  RTSP_PORT = 554,

  LAST_PRIVILEDGED_PORT_ID = 1024,

  // rtp audio bcast
  RTP_AUDIO_PORT = 5004,

  MDNS_PORT = 5353,

  LOCAL_WEB_PORT = 8443,

  // Session Announcement Protocol
  SAP_PORT_EVENT = 9875,

  UNKNOWN = 0xffff,
};
} // namespace iuring

#include <slogger/Error.hpp>

namespace iuring {
enum class UringFeature {
  UNKNOWN,
  IORING_OP_NOP,
  IORING_OP_READV,
  IORING_OP_WRITEV,
  IORING_OP_FSYNC,
  IORING_OP_READ_FIXED,
  IORING_OP_WRITE_FIXED,
  IORING_OP_POLL_ADD,
  IORING_OP_POLL_REMOVE,
  IORING_OP_SYNC_FILE_RANGE,
  IORING_OP_SENDMSG,
  IORING_OP_RECVMSG,
  IORING_OP_TIMEOUT,
  IORING_OP_TIMEOUT_REMOVE,
  IORING_OP_ACCEPT,
  IORING_OP_ASYNC_CANCEL,
  IORING_OP_LINK_TIMEOUT,
  IORING_OP_CONNECT,
  IORING_OP_FALLOCATE,
  IORING_OP_OPENAT,
  IORING_OP_CLOSE,
  IORING_OP_FILES_UPDATE,
  IORING_OP_STATX,
  IORING_OP_READ,
  IORING_OP_WRITE,
  IORING_OP_FADVISE,
  IORING_OP_MADVISE,
  IORING_OP_SEND,
  IORING_OP_RECV,
  IORING_OP_OPENAT2,
  IORING_OP_EPOLL_CTL,
  IORING_OP_SPLICE,
  IORING_OP_PROVIDE_BUFFERS,
  IORING_OP_REMOVE_BUFFERS,
  IORING_OP_TEE,
  IORING_OP_SHUTDOWN,
  IORING_OP_RENAMEAT,
  IORING_OP_UNLINKAT,
  IORING_OP_MKDIRAT,
  IORING_OP_SYMLINKAT,
  IORING_OP_LINKAT,
#if SUPPORT_LISTEN_IN_LIBURING
  IORING_OP_LISTEN,
#endif
};
} // namespace iuring