#pragma once

#include <Application.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <slogger/Logger.hpp>

#include <urtsched/ServiceBus.hpp>

#include <ptp/PtpHeaderV1.hpp>
#include <ptp/PtpHeaderV2.hpp>

#include <../tests/iuring_mocks.hpp>
#include <iuring/NetworkAdapter.hpp>

#include "audio_mocks.hpp"

#include <slogger/DefaultClockTimer.hpp>

#include <domainmodel/Device.hpp>
#include <domainmodel/GUID.hpp>

struct Context {
  int m_num_sends = 0;

  const std::string test_interface_ip4 = "1.2.3.4";
  const std::string interface_name = "eth0";

  time_utils::DefaultClockTimer m_timer;

  // during the unit tests we can spew a lot of debug msgs
  logging::DirectConsoleLogger m_logger{true, true, logging::LogOutput::CONSOLE};

  iuring::NetworkAdapter m_adapter{m_logger, interface_name, false};

  audio::AudioProfile m_profile = audio::default_ravenna_profile;

  std::shared_ptr<realtime::RealtimeKernel> rt_kernel =
      std::make_shared<realtime::RealtimeKernel>(m_timer, m_logger, "testcore");

  std::shared_ptr<model::Device> device = std::make_shared<model::Device>(
      m_logger, GUID::create_new_id(), "Test Device", "TestDevice1", nullptr,
      "RAVENNA");

  std::shared_ptr<ptp::mocks::PtpService> ptp =
      std::make_shared<ptp::mocks::PtpService>();

  std::shared_ptr<audio::mocks::AudioService> mock_audio_device =
      std::make_shared<audio::mocks::AudioService>(
          m_logger, m_profile, rt_kernel, device, ptp);

  std::shared_ptr<iuring::mocks::IOUring> mock_network =
      std::make_shared<iuring::mocks::IOUring>();
  uint16_t m_seq = 0;

  bool mdns_port_used = false;
  std::shared_ptr<iuring::mocks::Socket> mdns_port =
      std::make_shared<iuring::mocks::Socket>(iuring::SocketType::IPV4_UDP,
                                              iuring::SocketPortID::MDNS_PORT, m_logger,
                                              iuring::SocketKind::UNICAST_CLIENT_SOCKET, 42);

  bool web_port_used = false;
  std::shared_ptr<iuring::mocks::Socket> web_port =
      std::make_shared<iuring::mocks::Socket>(iuring::SocketType::IPV4_TCP,
                                              iuring::SocketPortID::UNENCRYPTED_WEB_PORT, m_logger,
                                              iuring::SocketKind::UNICAST_CLIENT_SOCKET, 42);

  bool rtp_port_used = false;
  std::shared_ptr<iuring::mocks::Socket> rtp_port =
      std::make_shared<iuring::mocks::Socket>(iuring::SocketType::IPV4_UDP,
                                              iuring::SocketPortID::RTP_AUDIO_PORT, m_logger,
                                              iuring::SocketKind::UNICAST_CLIENT_SOCKET, 42);

  bool event_port_used = false;
  std::shared_ptr<iuring::mocks::Socket> event_port =
      std::make_shared<iuring::mocks::Socket>(iuring::SocketType::IPV4_UDP,
                                              iuring::SocketPortID::PTP_PORT_EVENT, m_logger,
                                              iuring::SocketKind::UNICAST_CLIENT_SOCKET, 42);

  bool general_port_used = false;
  std::shared_ptr<iuring::mocks::Socket> general_port =
      std::make_shared<iuring::mocks::Socket>(iuring::SocketType::IPV4_UDP,
                                              iuring::SocketPortID::PTP_PORT_GENERAL, m_logger,
                                              iuring::SocketKind::UNICAST_CLIENT_SOCKET, 42);

  service::ServiceBus m_bus;

  enum class ReceiveSocketType {
    EVENT,
    GENERAL
  };

private:
  class ReceiveRequest {
  public:
    ReceiveRequest(ReceiveSocketType st, const std::vector<u_int8_t> &data,
                   const std::shared_ptr<iuring::mocks::Socket> &sock)
        : m_socket(sock), m_st(st), m_data(data) {
    }

    bool matches(ReceiveSocketType st) const {
      return !m_done && (st == m_st);
    }

    bool matches(const std::shared_ptr<iuring::IWorkItem> &wi) const {
      return m_socket == wi->get_socket();
    }

    bool is_done() const {
      return m_done;
    }

    size_t copy_out(msghdr *hdr) {
      static sockaddr_in tmp;

      size_t len = std::min(hdr->msg_iov->iov_len, m_data.size());
      memcpy(hdr->msg_iov->iov_base, m_data.data(), len);
      hdr->msg_namelen = sizeof(tmp);
      // hdr->msg_name = &tmp;
      m_done = true;
      return len;
    }

  private:
    std::shared_ptr<iuring::mocks::Socket> m_socket;
    ReceiveSocketType m_st;
    std::vector<u_int8_t> m_data;
    bool m_done = false;
  };

  std::vector<ReceiveRequest> todo;

  void inject_raw_packet(const ReceiveRequest &p) {
    todo.push_back(p);
  }

  template <typename T>
  void vec_append(std::vector<u_int8_t> &vec, const T &data) {
    uint8_t *raw = (uint8_t *)&data;
    for (size_t i = 0; i < sizeof(data); i++) {
      vec.push_back(raw[i]);
    }
  }

public:
  logging::ILogger &get_logger() {
    return m_logger;
  }

  void inject_received_packet(const ptp::v1::SyncMessageV1 &msg,
                              const std::shared_ptr<iuring::mocks::Socket> &sock) {
    const auto uuid = ptp::v1::convert_string_to_uuid(
        m_adapter.get_my_mac_address(), get_logger());
    ptp::v1::PtpHeaderV1 hdr(
        ptp::v1::PtpV1_PacketControlValue::PTP_SYNC_MESSAGE, m_seq++, uuid);

    std::vector<u_int8_t> data;
    vec_append(data, hdr);
    vec_append(data, msg);
    inject_raw_packet({ReceiveSocketType::EVENT, data, sock});
  }

  void inject_received_packet(const ptp::v1::FollowMessageV1 &msg,
                              const std::shared_ptr<iuring::mocks::Socket> &sock) {
    const auto uuid = ptp::v1::convert_string_to_uuid(
        m_adapter.get_my_mac_address(), get_logger());
    ptp::v1::PtpHeaderV1 hdr(
        ptp::v1::PtpV1_PacketControlValue::PTP_FOLLOWUP_MESSAGE, m_seq++,
        uuid);

    std::vector<u_int8_t> data;
    vec_append(data, hdr);
    vec_append(data, msg);
    inject_raw_packet({ReceiveSocketType::EVENT, data, sock});
  }

private:
  ssize_t try_pop_todo_receive_request(msghdr *hdr, ReceiveSocketType st) {
    EXPECT_GT(hdr->msg_iovlen, 0);
    EXPECT_GT(hdr->msg_iov[0].iov_len, 0);

    EXPECT_EQ(hdr->msg_iovlen, 1);

    for (auto &it : todo) {
      if (it.matches(st)) {
        return it.copy_out(hdr);
      }
    }
    return 0;
  }

  void garbage_collect_todo_receive_queue() {
    size_t i = 0;
    while (i < todo.size()) {
      if (todo[i].is_done()) {
        todo.erase(todo.begin() + i);
      } else {
        i++;
      }
    }
  }

  bool in_todo(const std::shared_ptr<iuring::IWorkItem> &wi) const {
    for (const auto &t : todo) {
      if (!t.is_done()) {
        if (t.matches(wi)) {
          return true;
        }
      }
    }
    return false;
  }

public:
  Context() {
    using testing::_;

    m_adapter.set_interface_ip4(iuring::IPAddress::parse("1.2.3.4").value());

    // mock_network->set_interface_ip4(test_interface_ip4);

    EXPECT_CALL(*mock_network, init());

    EXPECT_CALL(*mock_network, poll_completion_queues())
        .WillRepeatedly([this]() {
          garbage_collect_todo_receive_queue();
          return error::Error::OK;
        });

    EXPECT_CALL(*mock_network, submit(_))
        .WillRepeatedly([](iuring::IWorkItem &) {});

    EXPECT_CALL(*mock_network, submit_connect(_, _, _))
        .WillRepeatedly([](const std::shared_ptr<iuring::ISocket> &,
                           const iuring::IPAddress &, iuring::connect_callback_func_t handler) {
          iuring::ConnectResult res(0, iuring::IPAddress::parse("1.2.3.4").value());
          handler(res);
        });

    EXPECT_CALL(*mock_network, submit_recv(_, _))
        .WillRepeatedly([this](const std::shared_ptr<iuring::ISocket> &,
                               iuring::recv_callback_func_t handler) {
          const char *data = "";
          auto sa_opt = iuring::IPAddress::parse("1.2.3.4");
          assert(sa_opt.has_value());
          auto sa = *sa_opt;
          iuring::ReceivedMessage res((const uint8_t *)data, strlen(data), sa);
          auto action = handler(res);
          if (action == iuring::ReceivePostAction::RE_SUBMIT) {
            LOG_INFO(get_logger(), "test wants to resubmit");
          }
        });

    EXPECT_CALL(*mock_network, submit_accept(_, _))
        .WillRepeatedly([](const std::shared_ptr<iuring::ISocket> &,
                           iuring::accept_callback_func_t handler) {
          iuring::AcceptResult res;
          res.m_address = iuring::IPAddress::parse("1.2.3.4").value();
          handler(res);
        });

    EXPECT_CALL(*mock_network, submit_close(_, _))
        .WillRepeatedly([](const std::shared_ptr<iuring::ISocket> &,
                           iuring::close_callback_func_t handler) {
          iuring::CloseResult res(0);
          handler(res);
        });

    EXPECT_CALL(*general_port, mcast_bind());
    EXPECT_CALL(*event_port, mcast_bind());

    EXPECT_CALL(*general_port, join_multicast_group(_, _))
        .WillRepeatedly([]() {});
    EXPECT_CALL(*event_port, join_multicast_group(_, _))
        .WillRepeatedly([]() {});
    EXPECT_CALL(*rtp_port, join_multicast_group(_, _)).WillRepeatedly([]() {
    });
    EXPECT_CALL(*mdns_port, join_multicast_group(_, _))
        .WillRepeatedly([]() {});
  }
};

class BaseTest : public testing::Test {
public:
  void SetUp() override {}

  void TearDown() override {}

  time_utils::DefaultClockTimer m_timer;

  time_utils::DefaultClockTimer &get_timer() { return m_timer; }
};

class NetworkTest : public BaseTest {
public:
  void SetUp() override {
    m_context = std::make_shared<Context>();
  }

  void TearDown() override {
    m_context = nullptr;
  }

public:
  std::shared_ptr<Context> m_context;
};
#pragma once

#include <gmock/gmock.h>

#include <aoip/IRTP_Service.hpp>
#include <ptp/IPtpService.hpp>

#include <audio/AlsaAudioService.hpp>

namespace audio::mocks {
class RTP_Service : public IRTP_Service {
public:
  MOCK_METHOD(error::Error, init, (), (override));
  MOCK_METHOD(error::Error, finish, (), (override));

  MOCK_METHOD(void, add_receive_flow,
              (const std::shared_ptr<model::Flow> &flow), (override));
  MOCK_METHOD(void, remove_receive_flow,
              (const std::shared_ptr<model::Flow> &flow), (override));
  MOCK_METHOD(bool, already_have_flow,
              (const std::shared_ptr<model::Flow> &flow), (override));
};

class AudioService : public IAudioService {
public:
  AudioService(logging::ILogger &logger, AudioProfile &profile,
               const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
               std::shared_ptr<model::Device> device,
               const std::shared_ptr<ptp::IPtpService> &ptp_service)
      : IAudioService(logger, profile, rt_kernel, device, ptp_service) {
  }

  MOCK_METHOD(std::string, get_name, (), (const, override));
  MOCK_METHOD(std::string, get_description, (), (const, override));

  MOCK_METHOD(std::vector<std::string>, get_hw_input_channel_names, (),
              (const, override));
  MOCK_METHOD(std::vector<std::string>, get_hw_output_channel_names, (),
              (const, override));
};

class AlsaManager : public audio::IAlsaManager {
public:
  MOCK_METHOD(std::vector<std::shared_ptr<audio::AlsaAudioService>>,
              get_devices,
              (logging::ILogger & logger,
               const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
               settings::Configuration &config, model::ChannelMapper &mapper,
               const std::shared_ptr<ptp::IPtpService> &ptp_service),
              (override));
};

} // namespace audio::mocks

namespace ptp::mocks {
class PtpService : public ptp::IPtpService {
public:
  MOCK_METHOD(v2::clock_identity_t, get_v2_clk_id, (), (override));
  MOCK_METHOD(std::chrono::nanoseconds, get_time, (), (const, override));
};
} // namespace ptp::mocks
// Originally designed by Henry Schreiner
// https://github.com/CLIUtils/CLI11
//
// This is a standalone header file generated by MakeSingleHeader.py in CLI11/scripts
// from: v2.4.2
//
// CLI11 2.4.2 Copyright (c) 2017-2024 University of Cincinnati, developed by Henry
// Schreiner under NSF AWARD 1414736. All rights reserved.
//
// Redistribution and use in source and binary forms of CLI11, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

// Standard combined includes:
#include <algorithm>
#include <array>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#define CLI11_VERSION_MAJOR 2
#define CLI11_VERSION_MINOR 4
#define CLI11_VERSION_PATCH 2
#define CLI11_VERSION "2.4.2"

// The following version macro is very similar to the one in pybind11
#if !(defined(_MSC_VER) && __cplusplus == 199711L) && !defined(__INTEL_COMPILER)
#if __cplusplus >= 201402L
#define CLI11_CPP14
#if __cplusplus >= 201703L
#define CLI11_CPP17
#if __cplusplus > 201703L
#define CLI11_CPP20
#endif
#endif
#endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
// MSVC sets _MSVC_LANG rather than __cplusplus (supposedly until the standard is fully implemented)
// Unless you use the /Zc:__cplusplus flag on Visual Studio 2017 15.7 Preview 3 or newer
#if _MSVC_LANG >= 201402L
#define CLI11_CPP14
#if _MSVC_LANG > 201402L && _MSC_VER >= 1910
#define CLI11_CPP17
#if _MSVC_LANG > 201703L && _MSC_VER >= 1910
#define CLI11_CPP20
#endif
#endif
#endif
#endif

#if defined(CLI11_CPP14)
#define CLI11_DEPRECATED(reason) [[deprecated(reason)]]
#elif defined(_MSC_VER)
#define CLI11_DEPRECATED(reason) __declspec(deprecated(reason))
#else
#define CLI11_DEPRECATED(reason) __attribute__((deprecated(reason)))
#endif

// GCC < 10 doesn't ignore this in unevaluated contexts
#if !defined(CLI11_CPP17) || \
    (defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER) && __GNUC__ < 10 && __GNUC__ > 4)
#define CLI11_NODISCARD
#else
#define CLI11_NODISCARD [[nodiscard]]
#endif

/** detection of rtti */
#ifndef CLI11_USE_STATIC_RTTI
#if (defined(_HAS_STATIC_RTTI) && _HAS_STATIC_RTTI)
#define CLI11_USE_STATIC_RTTI 1
#elif defined(__cpp_rtti)
#if (defined(_CPPRTTI) && _CPPRTTI == 0)
#define CLI11_USE_STATIC_RTTI 1
#else
#define CLI11_USE_STATIC_RTTI 0
#endif
#elif (defined(__GCC_RTTI) && __GXX_RTTI)
#define CLI11_USE_STATIC_RTTI 0
#else
#define CLI11_USE_STATIC_RTTI 1
#endif
#endif

/** <filesystem> availability */
#if defined CLI11_CPP17 && defined __has_include && !defined CLI11_HAS_FILESYSTEM
#if __has_include(<filesystem>)
// Filesystem cannot be used if targeting macOS < 10.15
#if defined __MAC_OS_X_VERSION_MIN_REQUIRED && __MAC_OS_X_VERSION_MIN_REQUIRED < 101500
#define CLI11_HAS_FILESYSTEM 0
#elif defined(__wasi__)
// As of wasi-sdk-14, filesystem is not implemented
#define CLI11_HAS_FILESYSTEM 0
#else
#include <filesystem>
#if defined __cpp_lib_filesystem && __cpp_lib_filesystem >= 201703
#if defined _GLIBCXX_RELEASE && _GLIBCXX_RELEASE >= 9
#define CLI11_HAS_FILESYSTEM 1
#elif defined(__GLIBCXX__)
// if we are using gcc and Version <9 default to no filesystem
#define CLI11_HAS_FILESYSTEM 0
#else
#define CLI11_HAS_FILESYSTEM 1
#endif
#else
#define CLI11_HAS_FILESYSTEM 0
#endif
#endif
#endif
#endif

/** <codecvt> availability */
#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER) && __GNUC__ < 5
#define CLI11_HAS_CODECVT 0
#else
#define CLI11_HAS_CODECVT 1
#include <codecvt>
#endif

/** disable deprecations */
#if defined(__GNUC__) // GCC or clang
#define CLI11_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define CLI11_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")

#define CLI11_DIAGNOSTIC_IGNORE_DEPRECATED _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#elif defined(_MSC_VER)
#define CLI11_DIAGNOSTIC_PUSH __pragma(warning(push))
#define CLI11_DIAGNOSTIC_POP __pragma(warning(pop))

#define CLI11_DIAGNOSTIC_IGNORE_DEPRECATED __pragma(warning(disable : 4996))

#else
#define CLI11_DIAGNOSTIC_PUSH
#define CLI11_DIAGNOSTIC_POP

#define CLI11_DIAGNOSTIC_IGNORE_DEPRECATED

#endif

/** Inline macro **/
#ifdef CLI11_COMPILE
#define CLI11_INLINE
#else
#define CLI11_INLINE inline
#endif

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
#include <filesystem> // NOLINT(build/include)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifdef CLI11_CPP17
#include <string_view>
#endif // CLI11_CPP17

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
#include <filesystem>
#include <string_view> // NOLINT(build/include)
#endif                 // CLI11_HAS_FILESYSTEM

#if defined(_WIN32)
#if !(defined(_AMD64_) || defined(_X86_) || defined(_ARM_))
#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || \
    defined(_M_AMD64)
#define _AMD64_
#elif defined(i386) || defined(__i386) || defined(__i386__) || defined(__i386__) || defined(_M_IX86)
#define _X86_
#elif defined(__arm__) || defined(_M_ARM) || defined(_M_ARMT)
#define _ARM_
#elif defined(__aarch64__) || defined(_M_ARM64)
#define _ARM64_
#elif defined(_M_ARM64EC)
#define _ARM64EC_
#endif
#endif

// first
#ifndef NOMINMAX
// if NOMINMAX is already defined we don't want to mess with that either way
#define NOMINMAX
#include <windef.h>
#undef NOMINMAX
#else
#include <windef.h>
#endif

// second
#include <winbase.h>
// third
#include <processthreadsapi.h>
#include <shellapi.h>
#endif

namespace CLI {

/// Convert a wide string to a narrow string.
CLI11_INLINE std::string narrow(const std::wstring &str);
CLI11_INLINE std::string narrow(const wchar_t *str);
CLI11_INLINE std::string narrow(const wchar_t *str, std::size_t size);

/// Convert a narrow string to a wide string.
CLI11_INLINE std::wstring widen(const std::string &str);
CLI11_INLINE std::wstring widen(const char *str);
CLI11_INLINE std::wstring widen(const char *str, std::size_t size);

#ifdef CLI11_CPP17
CLI11_INLINE std::string narrow(std::wstring_view str);
CLI11_INLINE std::wstring widen(std::string_view str);
#endif // CLI11_CPP17

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
/// Convert a char-string to a native path correctly.
CLI11_INLINE std::filesystem::path to_path(std::string_view str);
#endif // CLI11_HAS_FILESYSTEM

namespace detail {

#if !CLI11_HAS_CODECVT
/// Attempt to set one of the acceptable unicode locales for conversion
CLI11_INLINE void set_unicode_locale() {
  static const std::array<const char *, 3> unicode_locales{{"C.UTF-8", "en_US.UTF-8", ".UTF-8"}};

  for (const auto &locale_name : unicode_locales) {
    if (std::setlocale(LC_ALL, locale_name) != nullptr) {
      return;
    }
  }
  throw std::runtime_error("CLI::narrow: could not set locale to C.UTF-8");
}

template <typename F>
struct scope_guard_t {
  F closure;

  explicit scope_guard_t(F closure_) : closure(closure_) {}
  ~scope_guard_t() { closure(); }
};

template <typename F>
CLI11_NODISCARD CLI11_INLINE scope_guard_t<F> scope_guard(F &&closure) {
  return scope_guard_t<F>{std::forward<F>(closure)};
}

#endif // !CLI11_HAS_CODECVT

CLI11_DIAGNOSTIC_PUSH
CLI11_DIAGNOSTIC_IGNORE_DEPRECATED

CLI11_INLINE std::string narrow_impl(const wchar_t *str, std::size_t str_size) {
#if CLI11_HAS_CODECVT
#ifdef _WIN32
  return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().to_bytes(str, str + str_size);

#else
  return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(str, str + str_size);

#endif // _WIN32
#else  // CLI11_HAS_CODECVT
  (void)str_size;
  std::mbstate_t state = std::mbstate_t();
  const wchar_t *it = str;

  std::string old_locale = std::setlocale(LC_ALL, nullptr);
  auto sg = scope_guard([&] { std::setlocale(LC_ALL, old_locale.c_str()); });
  set_unicode_locale();

  std::size_t new_size = std::wcsrtombs(nullptr, &it, 0, &state);
  if (new_size == static_cast<std::size_t>(-1)) {
    throw std::runtime_error("CLI::narrow: conversion error in std::wcsrtombs at offset " +
                             std::to_string(it - str));
  }
  std::string result(new_size, '\0');
  std::wcsrtombs(const_cast<char *>(result.data()), &str, new_size, &state);

  return result;

#endif // CLI11_HAS_CODECVT
}

CLI11_INLINE std::wstring widen_impl(const char *str, std::size_t str_size) {
#if CLI11_HAS_CODECVT
#ifdef _WIN32
  return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().from_bytes(str, str + str_size);

#else
  return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(str, str + str_size);

#endif // _WIN32
#else  // CLI11_HAS_CODECVT
  (void)str_size;
  std::mbstate_t state = std::mbstate_t();
  const char *it = str;

  std::string old_locale = std::setlocale(LC_ALL, nullptr);
  auto sg = scope_guard([&] { std::setlocale(LC_ALL, old_locale.c_str()); });
  set_unicode_locale();

  std::size_t new_size = std::mbsrtowcs(nullptr, &it, 0, &state);
  if (new_size == static_cast<std::size_t>(-1)) {
    throw std::runtime_error("CLI::widen: conversion error in std::mbsrtowcs at offset " +
                             std::to_string(it - str));
  }
  std::wstring result(new_size, L'\0');
  std::mbsrtowcs(const_cast<wchar_t *>(result.data()), &str, new_size, &state);

  return result;

#endif // CLI11_HAS_CODECVT
}

CLI11_DIAGNOSTIC_POP

} // namespace detail

CLI11_INLINE std::string narrow(const wchar_t *str, std::size_t str_size) { return detail::narrow_impl(str, str_size); }
CLI11_INLINE std::string narrow(const std::wstring &str) { return detail::narrow_impl(str.data(), str.size()); }
// Flawfinder: ignore
CLI11_INLINE std::string narrow(const wchar_t *str) { return detail::narrow_impl(str, std::wcslen(str)); }

CLI11_INLINE std::wstring widen(const char *str, std::size_t str_size) { return detail::widen_impl(str, str_size); }
CLI11_INLINE std::wstring widen(const std::string &str) { return detail::widen_impl(str.data(), str.size()); }
// Flawfinder: ignore
CLI11_INLINE std::wstring widen(const char *str) { return detail::widen_impl(str, std::strlen(str)); }

#ifdef CLI11_CPP17
CLI11_INLINE std::string narrow(std::wstring_view str) { return detail::narrow_impl(str.data(), str.size()); }
CLI11_INLINE std::wstring widen(std::string_view str) { return detail::widen_impl(str.data(), str.size()); }
#endif // CLI11_CPP17

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
CLI11_INLINE std::filesystem::path to_path(std::string_view str) {
  return std::filesystem::path{
#ifdef _WIN32
      widen(str)
#else
      str
#endif // _WIN32
  };
}
#endif // CLI11_HAS_FILESYSTEM

namespace detail {
#ifdef _WIN32
/// Decode and return UTF-8 argv from GetCommandLineW.
CLI11_INLINE std::vector<std::string> compute_win32_argv();
#endif
} // namespace detail

namespace detail {

#ifdef _WIN32
CLI11_INLINE std::vector<std::string> compute_win32_argv() {
  std::vector<std::string> result;
  int argc = 0;

  auto deleter = [](wchar_t **ptr) { LocalFree(ptr); };
  // NOLINTBEGIN(*-avoid-c-arrays)
  auto wargv = std::unique_ptr<wchar_t *[], decltype(deleter)>(CommandLineToArgvW(GetCommandLineW(), &argc), deleter);
  // NOLINTEND(*-avoid-c-arrays)

  if (wargv == nullptr) {
    throw std::runtime_error("CommandLineToArgvW failed with code " + std::to_string(GetLastError()));
  }

  result.reserve(static_cast<size_t>(argc));
  for (size_t i = 0; i < static_cast<size_t>(argc); ++i) {
    result.push_back(narrow(wargv[i]));
  }

  return result;
}
#endif

} // namespace detail

/// Include the items in this namespace to get free conversion of enums to/from streams.
/// (This is available inside CLI as well, so CLI11 will use this without a using statement).
namespace enums {

/// output streaming for enumerations
template <typename T, typename = typename std::enable_if<std::is_enum<T>::value>::type>
std::ostream &operator<<(std::ostream &in, const T &item) {
  // make sure this is out of the detail namespace otherwise it won't be found when needed
  return in << static_cast<typename std::underlying_type<T>::type>(item);
}

} // namespace enums

/// Export to CLI namespace
using enums::operator<<;

namespace detail {
/// a constant defining an expected max vector size defined to be a big number that could be multiplied by 4 and not
/// produce overflow for some expected uses
constexpr int expected_max_vector_size{1 << 29};
// Based on http://stackoverflow.com/questions/236129/split-a-string-in-c
/// Split a string by a delim
CLI11_INLINE std::vector<std::string> split(const std::string &s, char delim);

/// Simple function to join a string
template <typename T>
std::string join(const T &v, std::string delim = ",") {
  std::ostringstream s;
  auto beg = std::begin(v);
  auto end = std::end(v);
  if (beg != end)
    s << *beg++;
  while (beg != end) {
    s << delim << *beg++;
  }
  return s.str();
}

/// Simple function to join a string from processed elements
template <typename T,
          typename Callable,
          typename = typename std::enable_if<!std::is_constructible<std::string, Callable>::value>::type>
std::string join(const T &v, Callable func, std::string delim = ",") {
  std::ostringstream s;
  auto beg = std::begin(v);
  auto end = std::end(v);
  auto loc = s.tellp();
  while (beg != end) {
    auto nloc = s.tellp();
    if (nloc > loc) {
      s << delim;
      loc = nloc;
    }
    s << func(*beg++);
  }
  return s.str();
}

/// Join a string in reverse order
template <typename T>
std::string rjoin(const T &v, std::string delim = ",") {
  std::ostringstream s;
  for (std::size_t start = 0; start < v.size(); start++) {
    if (start > 0)
      s << delim;
    s << v[v.size() - start - 1];
  }
  return s.str();
}

// Based roughly on http://stackoverflow.com/questions/25829143/c-trim-whitespace-from-a-string

/// Trim whitespace from left of string
CLI11_INLINE std::string &ltrim(std::string &str);

/// Trim anything from left of string
CLI11_INLINE std::string &ltrim(std::string &str, const std::string &filter);

/// Trim whitespace from right of string
CLI11_INLINE std::string &rtrim(std::string &str);

/// Trim anything from right of string
CLI11_INLINE std::string &rtrim(std::string &str, const std::string &filter);

/// Trim whitespace from string
inline std::string &trim(std::string &str) { return ltrim(rtrim(str)); }

/// Trim anything from string
inline std::string &trim(std::string &str, const std::string filter) { return ltrim(rtrim(str, filter), filter); }

/// Make a copy of the string and then trim it
inline std::string trim_copy(const std::string &str) {
  std::string s = str;
  return trim(s);
}

/// remove quotes at the front and back of a string either '"' or '\''
CLI11_INLINE std::string &remove_quotes(std::string &str);

/// remove quotes from all elements of a string vector and process escaped components
CLI11_INLINE void remove_quotes(std::vector<std::string> &args);

/// Add a leader to the beginning of all new lines (nothing is added
/// at the start of the first line). `"; "` would be for ini files
///
/// Can't use Regex, or this would be a subs.
CLI11_INLINE std::string fix_newlines(const std::string &leader, std::string input);

/// Make a copy of the string and then trim it, any filter string can be used (any char in string is filtered)
inline std::string trim_copy(const std::string &str, const std::string &filter) {
  std::string s = str;
  return trim(s, filter);
}
/// Print a two part "help" string
CLI11_INLINE std::ostream &
format_help(std::ostream &out, std::string name, const std::string &description, std::size_t wid);

/// Print subcommand aliases
CLI11_INLINE std::ostream &format_aliases(std::ostream &out, const std::vector<std::string> &aliases, std::size_t wid);

/// Verify the first character of an option
/// - is a trigger character, ! has special meaning and new lines would just be annoying to deal with
template <typename T>
bool valid_first_char(T c) {
  return ((c != '-') && (static_cast<unsigned char>(c) > 33)); // space and '!' not allowed
}

/// Verify following characters of an option
template <typename T>
bool valid_later_char(T c) {
  // = and : are value separators, { has special meaning for option defaults,
  // and control codes other than tab would just be annoying to deal with in many places allowing space here has too
  // much potential for inadvertent entry errors and bugs
  return ((c != '=') && (c != ':') && (c != '{') && ((static_cast<unsigned char>(c) > 32) || c == '\t'));
}

/// Verify an option/subcommand name
CLI11_INLINE bool valid_name_string(const std::string &str);

/// Verify an app name
inline bool valid_alias_name_string(const std::string &str) {
  static const std::string badChars(std::string("\n") + '\0');
  return (str.find_first_of(badChars) == std::string::npos);
}

/// check if a string is a container segment separator (empty or "%%")
inline bool is_separator(const std::string &str) {
  static const std::string sep("%%");
  return (str.empty() || str == sep);
}

/// Verify that str consists of letters only
inline bool isalpha(const std::string &str) {
  return std::all_of(str.begin(), str.end(), [](char c) { return std::isalpha(c, std::locale()); });
}

/// Return a lower case version of a string
inline std::string to_lower(std::string str) {
  std::transform(std::begin(str), std::end(str), std::begin(str), [](const std::string::value_type &x) {
    return std::tolower(x, std::locale());
  });
  return str;
}

/// remove underscores from a string
inline std::string remove_underscore(std::string str) {
  str.erase(std::remove(std::begin(str), std::end(str), '_'), std::end(str));
  return str;
}

/// Find and replace a substring with another substring
CLI11_INLINE std::string find_and_replace(std::string str, std::string from, std::string to);

/// check if the flag definitions has possible false flags
inline bool has_default_flag_values(const std::string &flags) {
  return (flags.find_first_of("{!") != std::string::npos);
}

CLI11_INLINE void remove_default_flag_values(std::string &flags);

/// Check if a string is a member of a list of strings and optionally ignore case or ignore underscores
CLI11_INLINE std::ptrdiff_t find_member(std::string name,
                                        const std::vector<std::string> names,
                                        bool ignore_case = false,
                                        bool ignore_underscore = false);

/// Find a trigger string and call a modify callable function that takes the current string and starting position of the
/// trigger and returns the position in the string to search for the next trigger string
template <typename Callable>
inline std::string find_and_modify(std::string str, std::string trigger, Callable modify) {
  std::size_t start_pos = 0;
  while ((start_pos = str.find(trigger, start_pos)) != std::string::npos) {
    start_pos = modify(str, start_pos);
  }
  return str;
}

/// close a sequence of characters indicated by a closure character.  Brackets allows sub sequences
/// recognized bracket sequences include "'`[(<{  other closure characters are assumed to be literal strings
CLI11_INLINE std::size_t close_sequence(const std::string &str, std::size_t start, char closure_char);

/// Split a string '"one two" "three"' into 'one two', 'three'
/// Quote characters can be ` ' or " or bracket characters [{(< with matching to the matching bracket
CLI11_INLINE std::vector<std::string> split_up(std::string str, char delimiter = '\0');

/// get the value of an environmental variable or empty string if empty
CLI11_INLINE std::string get_environment_value(const std::string &env_name);

/// This function detects an equal or colon followed by an escaped quote after an argument
/// then modifies the string to replace the equality with a space.  This is needed
/// to allow the split up function to work properly and is intended to be used with the find_and_modify function
/// the return value is the offset+1 which is required by the find_and_modify function.
CLI11_INLINE std::size_t escape_detect(std::string &str, std::size_t offset);

/// @brief  detect if a string has escapable characters
/// @param str the string to do the detection on
/// @return true if the string has escapable characters
CLI11_INLINE bool has_escapable_character(const std::string &str);

/// @brief escape all escapable characters
/// @param str the string to escape
/// @return a string with the escapble characters escaped with '\'
CLI11_INLINE std::string add_escaped_characters(const std::string &str);

/// @brief replace the escaped characters with their equivalent
CLI11_INLINE std::string remove_escaped_characters(const std::string &str);

/// generate a string with all non printable characters escaped to hex codes
CLI11_INLINE std::string binary_escape_string(const std::string &string_to_escape);

CLI11_INLINE bool is_binary_escaped_string(const std::string &escaped_string);

/// extract an escaped binary_string
CLI11_INLINE std::string extract_binary_string(const std::string &escaped_string);

/// process a quoted string, remove the quotes and if appropriate handle escaped characters
CLI11_INLINE bool process_quoted_string(std::string &str, char string_char = '\"', char literal_char = '\'');

} // namespace detail

namespace detail {
CLI11_INLINE std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  // Check to see if empty string, give consistent result
  if (s.empty()) {
    elems.emplace_back();
  } else {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
      elems.push_back(item);
    }
  }
  return elems;
}

CLI11_INLINE std::string &ltrim(std::string &str) {
  auto it = std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace<char>(ch, std::locale()); });
  str.erase(str.begin(), it);
  return str;
}

CLI11_INLINE std::string &ltrim(std::string &str, const std::string &filter) {
  auto it = std::find_if(str.begin(), str.end(), [&filter](char ch) { return filter.find(ch) == std::string::npos; });
  str.erase(str.begin(), it);
  return str;
}

CLI11_INLINE std::string &rtrim(std::string &str) {
  auto it = std::find_if(str.rbegin(), str.rend(), [](char ch) { return !std::isspace<char>(ch, std::locale()); });
  str.erase(it.base(), str.end());
  return str;
}

CLI11_INLINE std::string &rtrim(std::string &str, const std::string &filter) {
  auto it =
      std::find_if(str.rbegin(), str.rend(), [&filter](char ch) { return filter.find(ch) == std::string::npos; });
  str.erase(it.base(), str.end());
  return str;
}

CLI11_INLINE std::string &remove_quotes(std::string &str) {
  if (str.length() > 1 && (str.front() == '"' || str.front() == '\'' || str.front() == '`')) {
    if (str.front() == str.back()) {
      str.pop_back();
      str.erase(str.begin(), str.begin() + 1);
    }
  }
  return str;
}

CLI11_INLINE std::string &remove_outer(std::string &str, char key) {
  if (str.length() > 1 && (str.front() == key)) {
    if (str.front() == str.back()) {
      str.pop_back();
      str.erase(str.begin(), str.begin() + 1);
    }
  }
  return str;
}

CLI11_INLINE std::string fix_newlines(const std::string &leader, std::string input) {
  std::string::size_type n = 0;
  while (n != std::string::npos && n < input.size()) {
    n = input.find('\n', n);
    if (n != std::string::npos) {
      input = input.substr(0, n + 1) + leader + input.substr(n + 1);
      n += leader.size();
    }
  }
  return input;
}

CLI11_INLINE std::ostream &
format_help(std::ostream &out, std::string name, const std::string &description, std::size_t wid) {
  name = "  " + name;
  out << std::setw(static_cast<int>(wid)) << std::left << name;
  if (!description.empty()) {
    if (name.length() >= wid)
      out << "\n"
          << std::setw(static_cast<int>(wid)) << "";
    for (const char c : description) {
      out.put(c);
      if (c == '\n') {
        out << std::setw(static_cast<int>(wid)) << "";
      }
    }
  }
  out << "\n";
  return out;
}

CLI11_INLINE std::ostream &format_aliases(std::ostream &out, const std::vector<std::string> &aliases, std::size_t wid) {
  if (!aliases.empty()) {
    out << std::setw(static_cast<int>(wid)) << "     aliases: ";
    bool front = true;
    for (const auto &alias : aliases) {
      if (!front) {
        out << ", ";
      } else {
        front = false;
      }
      out << detail::fix_newlines("              ", alias);
    }
    out << "\n";
  }
  return out;
}

CLI11_INLINE bool valid_name_string(const std::string &str) {
  if (str.empty() || !valid_first_char(str[0])) {
    return false;
  }
  auto e = str.end();
  for (auto c = str.begin() + 1; c != e; ++c)
    if (!valid_later_char(*c))
      return false;
  return true;
}

CLI11_INLINE std::string find_and_replace(std::string str, std::string from, std::string to) {

  std::size_t start_pos = 0;

  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }

  return str;
}

CLI11_INLINE void remove_default_flag_values(std::string &flags) {
  auto loc = flags.find_first_of('{', 2);
  while (loc != std::string::npos) {
    auto finish = flags.find_first_of("},", loc + 1);
    if ((finish != std::string::npos) && (flags[finish] == '}')) {
      flags.erase(flags.begin() + static_cast<std::ptrdiff_t>(loc),
                  flags.begin() + static_cast<std::ptrdiff_t>(finish) + 1);
    }
    loc = flags.find_first_of('{', loc + 1);
  }
  flags.erase(std::remove(flags.begin(), flags.end(), '!'), flags.end());
}

CLI11_INLINE std::ptrdiff_t
find_member(std::string name, const std::vector<std::string> names, bool ignore_case, bool ignore_underscore) {
  auto it = std::end(names);
  if (ignore_case) {
    if (ignore_underscore) {
      name = detail::to_lower(detail::remove_underscore(name));
      it = std::find_if(std::begin(names), std::end(names), [&name](std::string local_name) {
        return detail::to_lower(detail::remove_underscore(local_name)) == name;
      });
    } else {
      name = detail::to_lower(name);
      it = std::find_if(std::begin(names), std::end(names), [&name](std::string local_name) {
        return detail::to_lower(local_name) == name;
      });
    }

  } else if (ignore_underscore) {
    name = detail::remove_underscore(name);
    it = std::find_if(std::begin(names), std::end(names), [&name](std::string local_name) {
      return detail::remove_underscore(local_name) == name;
    });
  } else {
    it = std::find(std::begin(names), std::end(names), name);
  }

  return (it != std::end(names)) ? (it - std::begin(names)) : (-1);
}

static const std::string escapedChars("\b\t\n\f\r\"\\");
static const std::string escapedCharsCode("btnfr\"\\");
static const std::string bracketChars{"\"'`[(<{"};
static const std::string matchBracketChars("\"'`])>}");

CLI11_INLINE bool has_escapable_character(const std::string &str) {
  return (str.find_first_of(escapedChars) != std::string::npos);
}

CLI11_INLINE std::string add_escaped_characters(const std::string &str) {
  std::string out;
  out.reserve(str.size() + 4);
  for (char s : str) {
    auto sloc = escapedChars.find_first_of(s);
    if (sloc != std::string::npos) {
      out.push_back('\\');
      out.push_back(escapedCharsCode[sloc]);
    } else {
      out.push_back(s);
    }
  }
  return out;
}

CLI11_INLINE std::uint32_t hexConvert(char hc) {
  int hcode{0};
  if (hc >= '0' && hc <= '9') {
    hcode = (hc - '0');
  } else if (hc >= 'A' && hc <= 'F') {
    hcode = (hc - 'A' + 10);
  } else if (hc >= 'a' && hc <= 'f') {
    hcode = (hc - 'a' + 10);
  } else {
    hcode = -1;
  }
  return static_cast<uint32_t>(hcode);
}

CLI11_INLINE char make_char(std::uint32_t code) { return static_cast<char>(static_cast<unsigned char>(code)); }

CLI11_INLINE void append_codepoint(std::string &str, std::uint32_t code) {
  if (code < 0x80) { // ascii code equivalent
    str.push_back(static_cast<char>(code));
  } else if (code < 0x800) { // \u0080 to \u07FF
    // 110yyyyx 10xxxxxx; 0x3f == 0b0011'1111
    str.push_back(make_char(0xC0 | code >> 6));
    str.push_back(make_char(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) { // U+0800...U+FFFF
    if (0xD800 <= code && code <= 0xDFFF) {
      throw std::invalid_argument("[0xD800, 0xDFFF] are not valid UTF-8.");
    }
    // 1110yyyy 10yxxxxx 10xxxxxx
    str.push_back(make_char(0xE0 | code >> 12));
    str.push_back(make_char(0x80 | (code >> 6 & 0x3F)));
    str.push_back(make_char(0x80 | (code & 0x3F)));
  } else if (code < 0x110000) { // U+010000 ... U+10FFFF
    // 11110yyy 10yyxxxx 10xxxxxx 10xxxxxx
    str.push_back(make_char(0xF0 | code >> 18));
    str.push_back(make_char(0x80 | (code >> 12 & 0x3F)));
    str.push_back(make_char(0x80 | (code >> 6 & 0x3F)));
    str.push_back(make_char(0x80 | (code & 0x3F)));
  }
}

CLI11_INLINE std::string remove_escaped_characters(const std::string &str) {

  std::string out;
  out.reserve(str.size());
  for (auto loc = str.begin(); loc < str.end(); ++loc) {
    if (*loc == '\\') {
      if (str.end() - loc < 2) {
        throw std::invalid_argument("invalid escape sequence " + str);
      }
      auto ecloc = escapedCharsCode.find_first_of(*(loc + 1));
      if (ecloc != std::string::npos) {
        out.push_back(escapedChars[ecloc]);
        ++loc;
      } else if (*(loc + 1) == 'u') {
        // must have 4 hex characters
        if (str.end() - loc < 6) {
          throw std::invalid_argument("unicode sequence must have 4 hex codes " + str);
        }
        std::uint32_t code{0};
        std::uint32_t mplier{16 * 16 * 16};
        for (int ii = 2; ii < 6; ++ii) {
          std::uint32_t res = hexConvert(*(loc + ii));
          if (res > 0x0F) {
            throw std::invalid_argument("unicode sequence must have 4 hex codes " + str);
          }
          code += res * mplier;
          mplier = mplier / 16;
        }
        append_codepoint(out, code);
        loc += 5;
      } else if (*(loc + 1) == 'U') {
        // must have 8 hex characters
        if (str.end() - loc < 10) {
          throw std::invalid_argument("unicode sequence must have 8 hex codes " + str);
        }
        std::uint32_t code{0};
        std::uint32_t mplier{16 * 16 * 16 * 16 * 16 * 16 * 16};
        for (int ii = 2; ii < 10; ++ii) {
          std::uint32_t res = hexConvert(*(loc + ii));
          if (res > 0x0F) {
            throw std::invalid_argument("unicode sequence must have 8 hex codes " + str);
          }
          code += res * mplier;
          mplier = mplier / 16;
        }
        append_codepoint(out, code);
        loc += 9;
      } else if (*(loc + 1) == '0') {
        out.push_back('\0');
        ++loc;
      } else {
        throw std::invalid_argument(std::string("unrecognized escape sequence \\") + *(loc + 1) + " in " + str);
      }
    } else {
      out.push_back(*loc);
    }
  }
  return out;
}

CLI11_INLINE std::size_t close_string_quote(const std::string &str, std::size_t start, char closure_char) {
  std::size_t loc{0};
  for (loc = start + 1; loc < str.size(); ++loc) {
    if (str[loc] == closure_char) {
      break;
    }
    if (str[loc] == '\\') {
      // skip the next character for escaped sequences
      ++loc;
    }
  }
  return loc;
}

CLI11_INLINE std::size_t close_literal_quote(const std::string &str, std::size_t start, char closure_char) {
  auto loc = str.find_first_of(closure_char, start + 1);
  return (loc != std::string::npos ? loc : str.size());
}

CLI11_INLINE std::size_t close_sequence(const std::string &str, std::size_t start, char closure_char) {

  auto bracket_loc = matchBracketChars.find(closure_char);
  switch (bracket_loc) {
  case 0:
    return close_string_quote(str, start, closure_char);
  case 1:
  case 2:
  case std::string::npos:
    return close_literal_quote(str, start, closure_char);
  default:
    break;
  }

  std::string closures(1, closure_char);
  auto loc = start + 1;

  while (loc < str.size()) {
    if (str[loc] == closures.back()) {
      closures.pop_back();
      if (closures.empty()) {
        return loc;
      }
    }
    bracket_loc = bracketChars.find(str[loc]);
    if (bracket_loc != std::string::npos) {
      switch (bracket_loc) {
      case 0:
        loc = close_string_quote(str, loc, str[loc]);
        break;
      case 1:
      case 2:
        loc = close_literal_quote(str, loc, str[loc]);
        break;
      default:
        closures.push_back(matchBracketChars[bracket_loc]);
        break;
      }
    }
    ++loc;
  }
  if (loc > str.size()) {
    loc = str.size();
  }
  return loc;
}

CLI11_INLINE std::vector<std::string> split_up(std::string str, char delimiter) {

  auto find_ws = [delimiter](char ch) {
    return (delimiter == '\0') ? std::isspace<char>(ch, std::locale()) : (ch == delimiter);
  };
  trim(str);

  std::vector<std::string> output;
  while (!str.empty()) {
    if (bracketChars.find_first_of(str[0]) != std::string::npos) {
      auto bracketLoc = bracketChars.find_first_of(str[0]);
      auto end = close_sequence(str, 0, matchBracketChars[bracketLoc]);
      if (end >= str.size()) {
        output.push_back(std::move(str));
        str.clear();
      } else {
        output.push_back(str.substr(0, end + 1));
        if (end + 2 < str.size()) {
          str = str.substr(end + 2);
        } else {
          str.clear();
        }
      }

    } else {
      auto it = std::find_if(std::begin(str), std::end(str), find_ws);
      if (it != std::end(str)) {
        std::string value = std::string(str.begin(), it);
        output.push_back(value);
        str = std::string(it + 1, str.end());
      } else {
        output.push_back(str);
        str.clear();
      }
    }
    trim(str);
  }
  return output;
}

CLI11_INLINE std::size_t escape_detect(std::string &str, std::size_t offset) {
  auto next = str[offset + 1];
  if ((next == '\"') || (next == '\'') || (next == '`')) {
    auto astart = str.find_last_of("-/ \"\'`", offset - 1);
    if (astart != std::string::npos) {
      if (str[astart] == ((str[offset] == '=') ? '-' : '/'))
        str[offset] = ' '; // interpret this as a space so the split_up works properly
    }
  }
  return offset + 1;
}

CLI11_INLINE std::string binary_escape_string(const std::string &string_to_escape) {
  // s is our escaped output string
  std::string escaped_string{};
  // loop through all characters
  for (char c : string_to_escape) {
    // check if a given character is printable
    // the cast is necessary to avoid undefined behaviour
    if (isprint(static_cast<unsigned char>(c)) == 0) {
      std::stringstream stream;
      // if the character is not printable
      // we'll convert it to a hex string using a stringstream
      // note that since char is signed we have to cast it to unsigned first
      stream << std::hex << static_cast<unsigned int>(static_cast<unsigned char>(c));
      std::string code = stream.str();
      escaped_string += std::string("\\x") + (code.size() < 2 ? "0" : "") + code;

    } else {
      escaped_string.push_back(c);
    }
  }
  if (escaped_string != string_to_escape) {
    auto sqLoc = escaped_string.find('\'');
    while (sqLoc != std::string::npos) {
      escaped_string.replace(sqLoc, sqLoc + 1, "\\x27");
      sqLoc = escaped_string.find('\'');
    }
    escaped_string.insert(0, "'B\"(");
    escaped_string.push_back(')');
    escaped_string.push_back('"');
    escaped_string.push_back('\'');
  }
  return escaped_string;
}

CLI11_INLINE bool is_binary_escaped_string(const std::string &escaped_string) {
  size_t ssize = escaped_string.size();
  if (escaped_string.compare(0, 3, "B\"(") == 0 && escaped_string.compare(ssize - 2, 2, ")\"") == 0) {
    return true;
  }
  return (escaped_string.compare(0, 4, "'B\"(") == 0 && escaped_string.compare(ssize - 3, 3, ")\"'") == 0);
}

CLI11_INLINE std::string extract_binary_string(const std::string &escaped_string) {
  std::size_t start{0};
  std::size_t tail{0};
  size_t ssize = escaped_string.size();
  if (escaped_string.compare(0, 3, "B\"(") == 0 && escaped_string.compare(ssize - 2, 2, ")\"") == 0) {
    start = 3;
    tail = 2;
  } else if (escaped_string.compare(0, 4, "'B\"(") == 0 && escaped_string.compare(ssize - 3, 3, ")\"'") == 0) {
    start = 4;
    tail = 3;
  }

  if (start == 0) {
    return escaped_string;
  }
  std::string outstring;

  outstring.reserve(ssize - start - tail);
  std::size_t loc = start;
  while (loc < ssize - tail) {
    // ssize-2 to skip )" at the end
    if (escaped_string[loc] == '\\' && (escaped_string[loc + 1] == 'x' || escaped_string[loc + 1] == 'X')) {
      auto c1 = escaped_string[loc + 2];
      auto c2 = escaped_string[loc + 3];

      std::uint32_t res1 = hexConvert(c1);
      std::uint32_t res2 = hexConvert(c2);
      if (res1 <= 0x0F && res2 <= 0x0F) {
        loc += 4;
        outstring.push_back(static_cast<char>(res1 * 16 + res2));
        continue;
      }
    }
    outstring.push_back(escaped_string[loc]);
    ++loc;
  }
  return outstring;
}

CLI11_INLINE void remove_quotes(std::vector<std::string> &args) {
  for (auto &arg : args) {
    if (arg.front() == '\"' && arg.back() == '\"') {
      remove_quotes(arg);
      // only remove escaped for string arguments not literal strings
      arg = remove_escaped_characters(arg);
    } else {
      remove_quotes(arg);
    }
  }
}

CLI11_INLINE bool process_quoted_string(std::string &str, char string_char, char literal_char) {
  if (str.size() <= 1) {
    return false;
  }
  if (detail::is_binary_escaped_string(str)) {
    str = detail::extract_binary_string(str);
    return true;
  }
  if (str.front() == string_char && str.back() == string_char) {
    detail::remove_outer(str, string_char);
    if (str.find_first_of('\\') != std::string::npos) {
      str = detail::remove_escaped_characters(str);
    }
    return true;
  }
  if ((str.front() == literal_char || str.front() == '`') && str.back() == str.front()) {
    detail::remove_outer(str, str.front());
    return true;
  }
  return false;
}

std::string get_environment_value(const std::string &env_name) {
  char *buffer = nullptr;
  std::string ename_string;

#ifdef _MSC_VER
  // Windows version
  std::size_t sz = 0;
  if (_dupenv_s(&buffer, &sz, env_name.c_str()) == 0 && buffer != nullptr) {
    ename_string = std::string(buffer);
    free(buffer);
  }
#else
  // This also works on Windows, but gives a warning
  buffer = std::getenv(env_name.c_str());
  if (buffer != nullptr) {
    ename_string = std::string(buffer);
  }
#endif
  return ename_string;
}

} // namespace detail

// Use one of these on all error classes.
// These are temporary and are undef'd at the end of this file.
#define CLI11_ERROR_DEF(parent, name)                                                                              \
protected:                                                                                                         \
  name(std::string ename, std::string msg, int exit_code) : parent(std::move(ename), std::move(msg), exit_code) {} \
  name(std::string ename, std::string msg, ExitCodes exit_code)                                                    \
      : parent(std::move(ename), std::move(msg), exit_code) {}                                                     \
                                                                                                                   \
public:                                                                                                            \
  name(std::string msg, ExitCodes exit_code) : parent(#name, std::move(msg), exit_code) {}                         \
  name(std::string msg, int exit_code) : parent(#name, std::move(msg), exit_code) {}

// This is added after the one above if a class is used directly and builds its own message
#define CLI11_ERROR_SIMPLE(name) \
  explicit name(std::string msg) : name(#name, msg, ExitCodes::name) {}

/// These codes are part of every error in CLI. They can be obtained from e using e.exit_code or as a quick shortcut,
/// int values from e.get_error_code().
enum class ExitCodes {
  Success = 0,
  IncorrectConstruction = 100,
  BadNameString,
  OptionAlreadyAdded,
  FileError,
  ConversionError,
  ValidationError,
  RequiredError,
  RequiresError,
  ExcludesError,
  ExtrasError,
  ConfigError,
  InvalidError,
  HorribleError,
  OptionNotFound,
  ArgumentMismatch,
  BaseClass = 127
};

// Error definitions

/// @defgroup error_group Errors
/// @brief Errors thrown by CLI11
///
/// These are the errors that can be thrown. Some of them, like CLI::Success, are not really errors.
/// @{

/// All errors derive from this one
class Error : public std::runtime_error {
  int actual_exit_code;
  std::string error_name{"Error"};

public:
  CLI11_NODISCARD int get_exit_code() const { return actual_exit_code; }

  CLI11_NODISCARD std::string get_name() const { return error_name; }

  Error(std::string name, std::string msg, int exit_code = static_cast<int>(ExitCodes::BaseClass))
      : runtime_error(msg), actual_exit_code(exit_code), error_name(std::move(name)) {}

  Error(std::string name, std::string msg, ExitCodes exit_code) : Error(name, msg, static_cast<int>(exit_code)) {}
};

// Note: Using Error::Error constructors does not work on GCC 4.7

/// Construction errors (not in parsing)
class ConstructionError : public Error {
  CLI11_ERROR_DEF(Error, ConstructionError)
};

/// Thrown when an option is set to conflicting values (non-vector and multi args, for example)
class IncorrectConstruction : public ConstructionError {
  CLI11_ERROR_DEF(ConstructionError, IncorrectConstruction)
  CLI11_ERROR_SIMPLE(IncorrectConstruction)
  static IncorrectConstruction PositionalFlag(std::string name) {
    return IncorrectConstruction(name + ": Flags cannot be positional");
  }
  static IncorrectConstruction Set0Opt(std::string name) {
    return IncorrectConstruction(name + ": Cannot set 0 expected, use a flag instead");
  }
  static IncorrectConstruction SetFlag(std::string name) {
    return IncorrectConstruction(name + ": Cannot set an expected number for flags");
  }
  static IncorrectConstruction ChangeNotVector(std::string name) {
    return IncorrectConstruction(name + ": You can only change the expected arguments for vectors");
  }
  static IncorrectConstruction AfterMultiOpt(std::string name) {
    return IncorrectConstruction(
        name + ": You can't change expected arguments after you've changed the multi option policy!");
  }
  static IncorrectConstruction MissingOption(std::string name) {
    return IncorrectConstruction("Option " + name + " is not defined");
  }
  static IncorrectConstruction MultiOptionPolicy(std::string name) {
    return IncorrectConstruction(name + ": multi_option_policy only works for flags and exact value options");
  }
};

/// Thrown on construction of a bad name
class BadNameString : public ConstructionError {
  CLI11_ERROR_DEF(ConstructionError, BadNameString)
  CLI11_ERROR_SIMPLE(BadNameString)
  static BadNameString OneCharName(std::string name) { return BadNameString("Invalid one char name: " + name); }
  static BadNameString MissingDash(std::string name) {
    return BadNameString("Long names strings require 2 dashes " + name);
  }
  static BadNameString BadLongName(std::string name) { return BadNameString("Bad long name: " + name); }
  static BadNameString BadPositionalName(std::string name) {
    return BadNameString("Invalid positional Name: " + name);
  }
  static BadNameString DashesOnly(std::string name) {
    return BadNameString("Must have a name, not just dashes: " + name);
  }
  static BadNameString MultiPositionalNames(std::string name) {
    return BadNameString("Only one positional name allowed, remove: " + name);
  }
};

/// Thrown when an option already exists
class OptionAlreadyAdded : public ConstructionError {
  CLI11_ERROR_DEF(ConstructionError, OptionAlreadyAdded)
  explicit OptionAlreadyAdded(std::string name)
      : OptionAlreadyAdded(name + " is already added", ExitCodes::OptionAlreadyAdded) {}
  static OptionAlreadyAdded Requires(std::string name, std::string other) {
    return {name + " requires " + other, ExitCodes::OptionAlreadyAdded};
  }
  static OptionAlreadyAdded Excludes(std::string name, std::string other) {
    return {name + " excludes " + other, ExitCodes::OptionAlreadyAdded};
  }
};

// Parsing errors

/// Anything that can error in Parse
class ParseError : public Error {
  CLI11_ERROR_DEF(Error, ParseError)
};

// Not really "errors"

/// This is a successful completion on parsing, supposed to exit
class Success : public ParseError {
  CLI11_ERROR_DEF(ParseError, Success)
  Success() : Success("Successfully completed, should be caught and quit", ExitCodes::Success) {}
};

/// -h or --help on command line
class CallForHelp : public Success {
  CLI11_ERROR_DEF(Success, CallForHelp)
  CallForHelp() : CallForHelp("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

/// Usually something like --help-all on command line
class CallForAllHelp : public Success {
  CLI11_ERROR_DEF(Success, CallForAllHelp)
  CallForAllHelp()
      : CallForAllHelp("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

/// -v or --version on command line
class CallForVersion : public Success {
  CLI11_ERROR_DEF(Success, CallForVersion)
  CallForVersion()
      : CallForVersion("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

/// Does not output a diagnostic in CLI11_PARSE, but allows main() to return with a specific error code.
class RuntimeError : public ParseError {
  CLI11_ERROR_DEF(ParseError, RuntimeError)
  explicit RuntimeError(int exit_code = 1) : RuntimeError("Runtime error", exit_code) {}
};

/// Thrown when parsing an INI file and it is missing
class FileError : public ParseError {
  CLI11_ERROR_DEF(ParseError, FileError)
  CLI11_ERROR_SIMPLE(FileError)
  static FileError Missing(std::string name) { return FileError(name + " was not readable (missing?)"); }
};

/// Thrown when conversion call back fails, such as when an int fails to coerce to a string
class ConversionError : public ParseError {
  CLI11_ERROR_DEF(ParseError, ConversionError)
  CLI11_ERROR_SIMPLE(ConversionError)
  ConversionError(std::string member, std::string name)
      : ConversionError("The value " + member + " is not an allowed value for " + name) {}
  ConversionError(std::string name, std::vector<std::string> results)
      : ConversionError("Could not convert: " + name + " = " + detail::join(results)) {}
  static ConversionError TooManyInputsFlag(std::string name) {
    return ConversionError(name + ": too many inputs for a flag");
  }
  static ConversionError TrueFalse(std::string name) {
    return ConversionError(name + ": Should be true/false or a number");
  }
};

/// Thrown when validation of results fails
class ValidationError : public ParseError {
  CLI11_ERROR_DEF(ParseError, ValidationError)
  CLI11_ERROR_SIMPLE(ValidationError)
  explicit ValidationError(std::string name, std::string msg) : ValidationError(name + ": " + msg) {}
};

/// Thrown when a required option is missing
class RequiredError : public ParseError {
  CLI11_ERROR_DEF(ParseError, RequiredError)
  explicit RequiredError(std::string name) : RequiredError(name + " is required", ExitCodes::RequiredError) {}
  static RequiredError Subcommand(std::size_t min_subcom) {
    if (min_subcom == 1) {
      return RequiredError("A subcommand");
    }
    return {"Requires at least " + std::to_string(min_subcom) + " subcommands", ExitCodes::RequiredError};
  }
  static RequiredError
  Option(std::size_t min_option, std::size_t max_option, std::size_t used, const std::string &option_list) {
    if ((min_option == 1) && (max_option == 1) && (used == 0))
      return RequiredError("Exactly 1 option from [" + option_list + "]");
    if ((min_option == 1) && (max_option == 1) && (used > 1)) {
      return {"Exactly 1 option from [" + option_list + "] is required but " + std::to_string(used) +
                  " were given",
              ExitCodes::RequiredError};
    }
    if ((min_option == 1) && (used == 0))
      return RequiredError("At least 1 option from [" + option_list + "]");
    if (used < min_option) {
      return {"Requires at least " + std::to_string(min_option) + " options used but only " +
                  std::to_string(used) + " were given from [" + option_list + "]",
              ExitCodes::RequiredError};
    }
    if (max_option == 1)
      return {"Requires at most 1 options be given from [" + option_list + "]", ExitCodes::RequiredError};

    return {"Requires at most " + std::to_string(max_option) + " options be used but " + std::to_string(used) +
                " were given from [" + option_list + "]",
            ExitCodes::RequiredError};
  }
};

/// Thrown when the wrong number of arguments has been received
class ArgumentMismatch : public ParseError {
  CLI11_ERROR_DEF(ParseError, ArgumentMismatch)
  CLI11_ERROR_SIMPLE(ArgumentMismatch)
  ArgumentMismatch(std::string name, int expected, std::size_t received)
      : ArgumentMismatch(expected > 0 ? ("Expected exactly " + std::to_string(expected) + " arguments to " + name +
                                         ", got " + std::to_string(received))
                                      : ("Expected at least " + std::to_string(-expected) + " arguments to " + name +
                                         ", got " + std::to_string(received)),
                         ExitCodes::ArgumentMismatch) {}

  static ArgumentMismatch AtLeast(std::string name, int num, std::size_t received) {
    return ArgumentMismatch(name + ": At least " + std::to_string(num) + " required but received " +
                            std::to_string(received));
  }
  static ArgumentMismatch AtMost(std::string name, int num, std::size_t received) {
    return ArgumentMismatch(name + ": At Most " + std::to_string(num) + " required but received " +
                            std::to_string(received));
  }
  static ArgumentMismatch TypedAtLeast(std::string name, int num, std::string type) {
    return ArgumentMismatch(name + ": " + std::to_string(num) + " required " + type + " missing");
  }
  static ArgumentMismatch FlagOverride(std::string name) {
    return ArgumentMismatch(name + " was given a disallowed flag override");
  }
  static ArgumentMismatch PartialType(std::string name, int num, std::string type) {
    return ArgumentMismatch(name + ": " + type + " only partially specified: " + std::to_string(num) +
                            " required for each element");
  }
};

/// Thrown when a requires option is missing
class RequiresError : public ParseError {
  CLI11_ERROR_DEF(ParseError, RequiresError)
  RequiresError(std::string curname, std::string subname)
      : RequiresError(curname + " requires " + subname, ExitCodes::RequiresError) {}
};

/// Thrown when an excludes option is present
class ExcludesError : public ParseError {
  CLI11_ERROR_DEF(ParseError, ExcludesError)
  ExcludesError(std::string curname, std::string subname)
      : ExcludesError(curname + " excludes " + subname, ExitCodes::ExcludesError) {}
};

/// Thrown when too many positionals or options are found
class ExtrasError : public ParseError {
  CLI11_ERROR_DEF(ParseError, ExtrasError)
  explicit ExtrasError(std::vector<std::string> args)
      : ExtrasError((args.size() > 1 ? "The following arguments were not expected: "
                                     : "The following argument was not expected: ") +
                        detail::rjoin(args, " "),
                    ExitCodes::ExtrasError) {}
  ExtrasError(const std::string &name, std::vector<std::string> args)
      : ExtrasError(name,
                    (args.size() > 1 ? "The following arguments were not expected: "
                                     : "The following argument was not expected: ") +
                        detail::rjoin(args, " "),
                    ExitCodes::ExtrasError) {}
};

/// Thrown when extra values are found in an INI file
class ConfigError : public ParseError {
  CLI11_ERROR_DEF(ParseError, ConfigError)
  CLI11_ERROR_SIMPLE(ConfigError)
  static ConfigError Extras(std::string item) { return ConfigError("INI was not able to parse " + item); }
  static ConfigError NotConfigurable(std::string item) {
    return ConfigError(item + ": This option is not allowed in a configuration file");
  }
};

/// Thrown when validation fails before parsing
class InvalidError : public ParseError {
  CLI11_ERROR_DEF(ParseError, InvalidError)
  explicit InvalidError(std::string name)
      : InvalidError(name + ": Too many positional arguments with unlimited expected args", ExitCodes::InvalidError) {
  }
};

/// This is just a safety check to verify selection and parsing match - you should not ever see it
/// Strings are directly added to this error, but again, it should never be seen.
class HorribleError : public ParseError {
  CLI11_ERROR_DEF(ParseError, HorribleError)
  CLI11_ERROR_SIMPLE(HorribleError)
};

// After parsing

/// Thrown when counting a non-existent option
class OptionNotFound : public Error {
  CLI11_ERROR_DEF(Error, OptionNotFound)
  explicit OptionNotFound(std::string name) : OptionNotFound(name + " not found", ExitCodes::OptionNotFound) {}
};

#undef CLI11_ERROR_DEF
#undef CLI11_ERROR_SIMPLE

/// @}

// Type tools

// Utilities for type enabling
namespace detail {
// Based generally on https://rmf.io/cxx11/almost-static-if
/// Simple empty scoped class
enum class enabler {};

/// An instance to use in EnableIf
constexpr enabler dummy = {};
} // namespace detail

/// A copy of enable_if_t from C++14, compatible with C++11.
///
/// We could check to see if C++14 is being used, but it does not hurt to redefine this
/// (even Google does this: https://github.com/google/skia/blob/main/include/private/SkTLogic.h)
/// It is not in the std namespace anyway, so no harm done.
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/// A copy of std::void_t from C++17 (helper for C++11 and C++14)
template <typename... Ts>
struct make_void {
  using type = void;
};

/// A copy of std::void_t from C++17 - same reasoning as enable_if_t, it does not hurt to redefine
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

/// A copy of std::conditional_t from C++14 - same reasoning as enable_if_t, it does not hurt to redefine
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;

/// Check to see if something is bool (fail check by default)
template <typename T>
struct is_bool : std::false_type {};

/// Check to see if something is bool (true if actually a bool)
template <>
struct is_bool<bool> : std::true_type {};

/// Check to see if something is a shared pointer
template <typename T>
struct is_shared_ptr : std::false_type {};

/// Check to see if something is a shared pointer (True if really a shared pointer)
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

/// Check to see if something is a shared pointer (True if really a shared pointer)
template <typename T>
struct is_shared_ptr<const std::shared_ptr<T>> : std::true_type {};

/// Check to see if something is copyable pointer
template <typename T>
struct is_copyable_ptr {
  static bool const value = is_shared_ptr<T>::value || std::is_pointer<T>::value;
};

/// This can be specialized to override the type deduction for IsMember.
template <typename T>
struct IsMemberType {
  using type = T;
};

/// The main custom type needed here is const char * should be a string.
template <>
struct IsMemberType<const char *> {
  using type = std::string;
};

namespace adl_detail {
/// Check for existence of user-supplied lexical_cast.
///
/// This struct has to be in a separate namespace so that it doesn't see our lexical_cast overloads in CLI::detail.
/// Standard says it shouldn't see them if it's defined before the corresponding lexical_cast declarations, but this
/// requires a working implementation of two-phase lookup, and not all compilers can boast that (msvc, ahem).
template <typename T, typename S = std::string>
class is_lexical_castable {
  template <typename TT, typename SS>
  static auto test(int) -> decltype(lexical_cast(std::declval<const SS &>(), std::declval<TT &>()), std::true_type());

  template <typename, typename>
  static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T, S>(0))::value;
};
} // namespace adl_detail

namespace detail {

// These are utilities for IsMember and other transforming objects

/// Handy helper to access the element_type generically. This is not part of is_copyable_ptr because it requires that
/// pointer_traits<T> be valid.

/// not a pointer
template <typename T, typename Enable = void>
struct element_type {
  using type = T;
};

template <typename T>
struct element_type<T, typename std::enable_if<is_copyable_ptr<T>::value>::type> {
  using type = typename std::pointer_traits<T>::element_type;
};

/// Combination of the element type and value type - remove pointer (including smart pointers) and get the value_type of
/// the container
template <typename T>
struct element_value_type {
  using type = typename element_type<T>::type::value_type;
};

/// Adaptor for set-like structure: This just wraps a normal container in a few utilities that do almost nothing.
template <typename T, typename _ = void>
struct pair_adaptor : std::false_type {
  using value_type = typename T::value_type;
  using first_type = typename std::remove_const<value_type>::type;
  using second_type = typename std::remove_const<value_type>::type;

  /// Get the first value (really just the underlying value)
  template <typename Q>
  static auto first(Q &&pair_value) -> decltype(std::forward<Q>(pair_value)) {
    return std::forward<Q>(pair_value);
  }
  /// Get the second value (really just the underlying value)
  template <typename Q>
  static auto second(Q &&pair_value) -> decltype(std::forward<Q>(pair_value)) {
    return std::forward<Q>(pair_value);
  }
};

/// Adaptor for map-like structure (true version, must have key_type and mapped_type).
/// This wraps a mapped container in a few utilities access it in a general way.
template <typename T>
struct pair_adaptor<
    T,
    conditional_t<false, void_t<typename T::value_type::first_type, typename T::value_type::second_type>, void>>
    : std::true_type {
  using value_type = typename T::value_type;
  using first_type = typename std::remove_const<typename value_type::first_type>::type;
  using second_type = typename std::remove_const<typename value_type::second_type>::type;

  /// Get the first value (really just the underlying value)
  template <typename Q>
  static auto first(Q &&pair_value) -> decltype(std::get<0>(std::forward<Q>(pair_value))) {
    return std::get<0>(std::forward<Q>(pair_value));
  }
  /// Get the second value (really just the underlying value)
  template <typename Q>
  static auto second(Q &&pair_value) -> decltype(std::get<1>(std::forward<Q>(pair_value))) {
    return std::get<1>(std::forward<Q>(pair_value));
  }
};

// Warning is suppressed due to "bug" in gcc<5.0 and gcc 7.0 with c++17 enabled that generates a Wnarrowing warning
// in the unevaluated context even if the function that was using this wasn't used.  The standard says narrowing in
// brace initialization shouldn't be allowed but for backwards compatibility gcc allows it in some contexts.  It is a
// little fuzzy what happens in template constructs and I think that was something GCC took a little while to work out.
// But regardless some versions of gcc generate a warning when they shouldn't from the following code so that should be
// suppressed
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
// check for constructibility from a specific type and copy assignable used in the parse detection
template <typename T, typename C>
class is_direct_constructible {
  template <typename TT, typename CC>
  static auto test(int, std::true_type) -> decltype(
// NVCC warns about narrowing conversions here
#ifdef __CUDACC__
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 2361
#else
#pragma diag_suppress 2361
#endif
#endif
      TT{std::declval<CC>()}
#ifdef __CUDACC__
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_default 2361
#else
#pragma diag_default 2361
#endif
#endif
      ,
      std::is_move_assignable<TT>());

  template <typename TT, typename CC>
  static auto test(int, std::false_type) -> std::false_type;

  template <typename, typename>
  static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T, C>(0, typename std::is_constructible<T, C>::type()))::value;
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// Check for output streamability
// Based on https://stackoverflow.com/questions/22758291/how-can-i-detect-if-a-type-can-be-streamed-to-an-stdostream

template <typename T, typename S = std::ostringstream>
class is_ostreamable {
  template <typename TT, typename SS>
  static auto test(int) -> decltype(std::declval<SS &>() << std::declval<TT>(), std::true_type());

  template <typename, typename>
  static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T, S>(0))::value;
};

/// Check for input streamability
template <typename T, typename S = std::istringstream>
class is_istreamable {
  template <typename TT, typename SS>
  static auto test(int) -> decltype(std::declval<SS &>() >> std::declval<TT &>(), std::true_type());

  template <typename, typename>
  static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T, S>(0))::value;
};

/// Check for complex
template <typename T>
class is_complex {
  template <typename TT>
  static auto test(int) -> decltype(std::declval<TT>().real(), std::declval<TT>().imag(), std::true_type());

  template <typename>
  static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/// Templated operation to get a value from a stream
template <typename T, enable_if_t<is_istreamable<T>::value, detail::enabler> = detail::dummy>
bool from_stream(const std::string &istring, T &obj) {
  std::istringstream is;
  is.str(istring);
  is >> obj;
  return !is.fail() && !is.rdbuf()->in_avail();
}

template <typename T, enable_if_t<!is_istreamable<T>::value, detail::enabler> = detail::dummy>
bool from_stream(const std::string & /*istring*/, T & /*obj*/) {
  return false;
}

// check to see if an object is a mutable container (fail by default)
template <typename T, typename _ = void>
struct is_mutable_container : std::false_type {};

/// type trait to test if a type is a mutable container meaning it has a value_type, it has an iterator, a clear, and
/// end methods and an insert function.  And for our purposes we exclude std::string and types that can be constructed
/// from a std::string
template <typename T>
struct is_mutable_container<
    T,
    conditional_t<false,
                  void_t<typename T::value_type,
                         decltype(std::declval<T>().end()),
                         decltype(std::declval<T>().clear()),
                         decltype(std::declval<T>().insert(std::declval<decltype(std::declval<T>().end())>(),
                                                           std::declval<const typename T::value_type &>()))>,
                  void>> : public conditional_t<std::is_constructible<T, std::string>::value ||
                                                    std::is_constructible<T, std::wstring>::value,
                                                std::false_type,
                                                std::true_type> {};

// check to see if an object is a mutable container (fail by default)
template <typename T, typename _ = void>
struct is_readable_container : std::false_type {};

/// type trait to test if a type is a container meaning it has a value_type, it has an iterator, a clear, and an end
/// methods and an insert function.  And for our purposes we exclude std::string and types that can be constructed from
/// a std::string
template <typename T>
struct is_readable_container<
    T,
    conditional_t<false, void_t<decltype(std::declval<T>().end()), decltype(std::declval<T>().begin())>, void>>
    : public std::true_type {};

// check to see if an object is a wrapper (fail by default)
template <typename T, typename _ = void>
struct is_wrapper : std::false_type {};

// check if an object is a wrapper (it has a value_type defined)
template <typename T>
struct is_wrapper<T, conditional_t<false, void_t<typename T::value_type>, void>> : public std::true_type {};

// Check for tuple like types, as in classes with a tuple_size type trait
template <typename S>
class is_tuple_like {
  template <typename SS>
  // static auto test(int)
  //     -> decltype(std::conditional<(std::tuple_size<SS>::value > 0), std::true_type, std::false_type>::type());
  static auto test(int) -> decltype(std::tuple_size<typename std::decay<SS>::type>::value, std::true_type{});
  template <typename>
  static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<S>(0))::value;
};

/// Convert an object to a string (directly forward if this can become a string)
template <typename T, enable_if_t<std::is_convertible<T, std::string>::value, detail::enabler> = detail::dummy>
auto to_string(T &&value) -> decltype(std::forward<T>(value)) {
  return std::forward<T>(value);
}

/// Construct a string from the object
template <typename T,
          enable_if_t<std::is_constructible<std::string, T>::value && !std::is_convertible<T, std::string>::value,
                      detail::enabler> = detail::dummy>
std::string to_string(const T &value) {
  return std::string(value); // NOLINT(google-readability-casting)
}

/// Convert an object to a string (streaming must be supported for that type)
template <typename T,
          enable_if_t<!std::is_convertible<std::string, T>::value && !std::is_constructible<std::string, T>::value &&
                          is_ostreamable<T>::value,
                      detail::enabler> = detail::dummy>
std::string to_string(T &&value) {
  std::stringstream stream;
  stream << value;
  return stream.str();
}

/// If conversion is not supported, return an empty string (streaming is not supported for that type)
template <typename T,
          enable_if_t<!std::is_constructible<std::string, T>::value && !is_ostreamable<T>::value &&
                          !is_readable_container<typename std::remove_const<T>::type>::value,
                      detail::enabler> = detail::dummy>
std::string to_string(T &&) {
  return {};
}

/// convert a readable container to a string
template <typename T,
          enable_if_t<!std::is_constructible<std::string, T>::value && !is_ostreamable<T>::value &&
                          is_readable_container<T>::value,
                      detail::enabler> = detail::dummy>
std::string to_string(T &&variable) {
  auto cval = variable.begin();
  auto end = variable.end();
  if (cval == end) {
    return {"{}"};
  }
  std::vector<std::string> defaults;
  while (cval != end) {
    defaults.emplace_back(CLI::detail::to_string(*cval));
    ++cval;
  }
  return {"[" + detail::join(defaults) + "]"};
}

/// special template overload
template <typename T1,
          typename T2,
          typename T,
          enable_if_t<std::is_same<T1, T2>::value, detail::enabler> = detail::dummy>
auto checked_to_string(T &&value) -> decltype(to_string(std::forward<T>(value))) {
  return to_string(std::forward<T>(value));
}

/// special template overload
template <typename T1,
          typename T2,
          typename T,
          enable_if_t<!std::is_same<T1, T2>::value, detail::enabler> = detail::dummy>
std::string checked_to_string(T &&) {
  return std::string{};
}
/// get a string as a convertible value for arithmetic types
template <typename T, enable_if_t<std::is_arithmetic<T>::value, detail::enabler> = detail::dummy>
std::string value_string(const T &value) {
  return std::to_string(value);
}
/// get a string as a convertible value for enumerations
template <typename T, enable_if_t<std::is_enum<T>::value, detail::enabler> = detail::dummy>
std::string value_string(const T &value) {
  return std::to_string(static_cast<typename std::underlying_type<T>::type>(value));
}
/// for other types just use the regular to_string function
template <typename T,
          enable_if_t<!std::is_enum<T>::value && !std::is_arithmetic<T>::value, detail::enabler> = detail::dummy>
auto value_string(const T &value) -> decltype(to_string(value)) {
  return to_string(value);
}

/// template to get the underlying value type if it exists or use a default
template <typename T, typename def, typename Enable = void>
struct wrapped_type {
  using type = def;
};

/// Type size for regular object types that do not look like a tuple
template <typename T, typename def>
struct wrapped_type<T, def, typename std::enable_if<is_wrapper<T>::value>::type> {
  using type = typename T::value_type;
};

/// This will only trigger for actual void type
template <typename T, typename Enable = void>
struct type_count_base {
  static const int value{0};
};

/// Type size for regular object types that do not look like a tuple
template <typename T>
struct type_count_base<T,
                       typename std::enable_if<!is_tuple_like<T>::value && !is_mutable_container<T>::value &&
                                               !std::is_void<T>::value>::type> {
  static constexpr int value{1};
};

/// the base tuple size
template <typename T>
struct type_count_base<T, typename std::enable_if<is_tuple_like<T>::value && !is_mutable_container<T>::value>::type> {
  static constexpr int value{std::tuple_size<T>::value};
};

/// Type count base for containers is the type_count_base of the individual element
template <typename T>
struct type_count_base<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
  static constexpr int value{type_count_base<typename T::value_type>::value};
};

/// Set of overloads to get the type size of an object

/// forward declare the subtype_count structure
template <typename T>
struct subtype_count;

/// forward declare the subtype_count_min structure
template <typename T>
struct subtype_count_min;

/// This will only trigger for actual void type
template <typename T, typename Enable = void>
struct type_count {
  static const int value{0};
};

/// Type size for regular object types that do not look like a tuple
template <typename T>
struct type_count<T,
                  typename std::enable_if<!is_wrapper<T>::value && !is_tuple_like<T>::value && !is_complex<T>::value &&
                                          !std::is_void<T>::value>::type> {
  static constexpr int value{1};
};

/// Type size for complex since it sometimes looks like a wrapper
template <typename T>
struct type_count<T, typename std::enable_if<is_complex<T>::value>::type> {
  static constexpr int value{2};
};

/// Type size of types that are wrappers,except complex and tuples(which can also be wrappers sometimes)
template <typename T>
struct type_count<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
  static constexpr int value{subtype_count<typename T::value_type>::value};
};

/// Type size of types that are wrappers,except containers complex and tuples(which can also be wrappers sometimes)
template <typename T>
struct type_count<T,
                  typename std::enable_if<is_wrapper<T>::value && !is_complex<T>::value && !is_tuple_like<T>::value &&
                                          !is_mutable_container<T>::value>::type> {
  static constexpr int value{type_count<typename T::value_type>::value};
};

/// 0 if the index > tuple size
template <typename T, std::size_t I>
constexpr typename std::enable_if<I == type_count_base<T>::value, int>::type tuple_type_size() {
  return 0;
}

/// Recursively generate the tuple type name
template <typename T, std::size_t I>
    constexpr typename std::enable_if < I<type_count_base<T>::value, int>::type tuple_type_size() {
  return subtype_count<typename std::tuple_element<I, T>::type>::value + tuple_type_size<T, I + 1>();
}

/// Get the type size of the sum of type sizes for all the individual tuple types
template <typename T>
struct type_count<T, typename std::enable_if<is_tuple_like<T>::value>::type> {
  static constexpr int value{tuple_type_size<T, 0>()};
};

/// definition of subtype count
template <typename T>
struct subtype_count {
  static constexpr int value{is_mutable_container<T>::value ? expected_max_vector_size : type_count<T>::value};
};

/// This will only trigger for actual void type
template <typename T, typename Enable = void>
struct type_count_min {
  static const int value{0};
};

/// Type size for regular object types that do not look like a tuple
template <typename T>
struct type_count_min<
    T,
    typename std::enable_if<!is_mutable_container<T>::value && !is_tuple_like<T>::value && !is_wrapper<T>::value &&
                            !is_complex<T>::value && !std::is_void<T>::value>::type> {
  static constexpr int value{type_count<T>::value};
};

/// Type size for complex since it sometimes looks like a wrapper
template <typename T>
struct type_count_min<T, typename std::enable_if<is_complex<T>::value>::type> {
  static constexpr int value{1};
};

/// Type size min of types that are wrappers,except complex and tuples(which can also be wrappers sometimes)
template <typename T>
struct type_count_min<
    T,
    typename std::enable_if<is_wrapper<T>::value && !is_complex<T>::value && !is_tuple_like<T>::value>::type> {
  static constexpr int value{subtype_count_min<typename T::value_type>::value};
};

/// 0 if the index > tuple size
template <typename T, std::size_t I>
constexpr typename std::enable_if<I == type_count_base<T>::value, int>::type tuple_type_size_min() {
  return 0;
}

/// Recursively generate the tuple type name
template <typename T, std::size_t I>
    constexpr typename std::enable_if < I<type_count_base<T>::value, int>::type tuple_type_size_min() {
  return subtype_count_min<typename std::tuple_element<I, T>::type>::value + tuple_type_size_min<T, I + 1>();
}

/// Get the type size of the sum of type sizes for all the individual tuple types
template <typename T>
struct type_count_min<T, typename std::enable_if<is_tuple_like<T>::value>::type> {
  static constexpr int value{tuple_type_size_min<T, 0>()};
};

/// definition of subtype count
template <typename T>
struct subtype_count_min {
  static constexpr int value{is_mutable_container<T>::value
                                 ? ((type_count<T>::value < expected_max_vector_size) ? type_count<T>::value : 0)
                                 : type_count_min<T>::value};
};

/// This will only trigger for actual void type
template <typename T, typename Enable = void>
struct expected_count {
  static const int value{0};
};

/// For most types the number of expected items is 1
template <typename T>
struct expected_count<T,
                      typename std::enable_if<!is_mutable_container<T>::value && !is_wrapper<T>::value &&
                                              !std::is_void<T>::value>::type> {
  static constexpr int value{1};
};
/// number of expected items in a vector
template <typename T>
struct expected_count<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
  static constexpr int value{expected_max_vector_size};
};

/// number of expected items in a vector
template <typename T>
struct expected_count<T, typename std::enable_if<!is_mutable_container<T>::value && is_wrapper<T>::value>::type> {
  static constexpr int value{expected_count<typename T::value_type>::value};
};

// Enumeration of the different supported categorizations of objects
enum class object_category : int {
  char_value = 1,
  integral_value = 2,
  unsigned_integral = 4,
  enumeration = 6,
  boolean_value = 8,
  floating_point = 10,
  number_constructible = 12,
  double_constructible = 14,
  integer_constructible = 16,
  // string like types
  string_assignable = 23,
  string_constructible = 24,
  wstring_assignable = 25,
  wstring_constructible = 26,
  other = 45,
  // special wrapper or container types
  wrapper_value = 50,
  complex_number = 60,
  tuple_value = 70,
  container_value = 80,

};

/// Set of overloads to classify an object according to type

/// some type that is not otherwise recognized
template <typename T, typename Enable = void>
struct classify_object {
  static constexpr object_category value{object_category::other};
};

/// Signed integers
template <typename T>
struct classify_object<
    T,
    typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, char>::value && std::is_signed<T>::value &&
                            !is_bool<T>::value && !std::is_enum<T>::value>::type> {
  static constexpr object_category value{object_category::integral_value};
};

/// Unsigned integers
template <typename T>
struct classify_object<T,
                       typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value &&
                                               !std::is_same<T, char>::value && !is_bool<T>::value>::type> {
  static constexpr object_category value{object_category::unsigned_integral};
};

/// single character values
template <typename T>
struct classify_object<T, typename std::enable_if<std::is_same<T, char>::value && !std::is_enum<T>::value>::type> {
  static constexpr object_category value{object_category::char_value};
};

/// Boolean values
template <typename T>
struct classify_object<T, typename std::enable_if<is_bool<T>::value>::type> {
  static constexpr object_category value{object_category::boolean_value};
};

/// Floats
template <typename T>
struct classify_object<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  static constexpr object_category value{object_category::floating_point};
};
#if defined _MSC_VER
// in MSVC wstring should take precedence if available this isn't as useful on other compilers due to the broader use of
// utf-8 encoding
#define WIDE_STRING_CHECK \
  !std::is_assignable<T &, std::wstring>::value && !std::is_constructible<T, std::wstring>::value
#define STRING_CHECK true
#else
#define WIDE_STRING_CHECK true
#define STRING_CHECK !std::is_assignable<T &, std::string>::value && !std::is_constructible<T, std::string>::value
#endif

/// String and similar direct assignment
template <typename T>
struct classify_object<
    T,
    typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value && WIDE_STRING_CHECK &&
                            std::is_assignable<T &, std::string>::value>::type> {
  static constexpr object_category value{object_category::string_assignable};
};

/// String and similar constructible and copy assignment
template <typename T>
struct classify_object<
    T,
    typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
                            !std::is_assignable<T &, std::string>::value && (type_count<T>::value == 1) &&
                            WIDE_STRING_CHECK && std::is_constructible<T, std::string>::value>::type> {
  static constexpr object_category value{object_category::string_constructible};
};

/// Wide strings
template <typename T>
struct classify_object<T,
                       typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
                                               STRING_CHECK && std::is_assignable<T &, std::wstring>::value>::type> {
  static constexpr object_category value{object_category::wstring_assignable};
};

template <typename T>
struct classify_object<
    T,
    typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
                            !std::is_assignable<T &, std::wstring>::value && (type_count<T>::value == 1) &&
                            STRING_CHECK && std::is_constructible<T, std::wstring>::value>::type> {
  static constexpr object_category value{object_category::wstring_constructible};
};

/// Enumerations
template <typename T>
struct classify_object<T, typename std::enable_if<std::is_enum<T>::value>::type> {
  static constexpr object_category value{object_category::enumeration};
};

template <typename T>
struct classify_object<T, typename std::enable_if<is_complex<T>::value>::type> {
  static constexpr object_category value{object_category::complex_number};
};

/// Handy helper to contain a bunch of checks that rule out many common types (integers, string like, floating point,
/// vectors, and enumerations
template <typename T>
struct uncommon_type {
  using type = typename std::conditional<
      !std::is_floating_point<T>::value && !std::is_integral<T>::value &&
          !std::is_assignable<T &, std::string>::value && !std::is_constructible<T, std::string>::value &&
          !std::is_assignable<T &, std::wstring>::value && !std::is_constructible<T, std::wstring>::value &&
          !is_complex<T>::value && !is_mutable_container<T>::value && !std::is_enum<T>::value,
      std::true_type,
      std::false_type>::type;
  static constexpr bool value = type::value;
};

/// wrapper type
template <typename T>
struct classify_object<T,
                       typename std::enable_if<(!is_mutable_container<T>::value && is_wrapper<T>::value &&
                                                !is_tuple_like<T>::value && uncommon_type<T>::value)>::type> {
  static constexpr object_category value{object_category::wrapper_value};
};

/// Assignable from double or int
template <typename T>
struct classify_object<T,
                       typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
                                               !is_wrapper<T>::value && is_direct_constructible<T, double>::value &&
                                               is_direct_constructible<T, int>::value>::type> {
  static constexpr object_category value{object_category::number_constructible};
};

/// Assignable from int
template <typename T>
struct classify_object<T,
                       typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
                                               !is_wrapper<T>::value && !is_direct_constructible<T, double>::value &&
                                               is_direct_constructible<T, int>::value>::type> {
  static constexpr object_category value{object_category::integer_constructible};
};

/// Assignable from double
template <typename T>
struct classify_object<T,
                       typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
                                               !is_wrapper<T>::value && is_direct_constructible<T, double>::value &&
                                               !is_direct_constructible<T, int>::value>::type> {
  static constexpr object_category value{object_category::double_constructible};
};

/// Tuple type
template <typename T>
struct classify_object<
    T,
    typename std::enable_if<is_tuple_like<T>::value &&
                            ((type_count<T>::value >= 2 && !is_wrapper<T>::value) ||
                             (uncommon_type<T>::value && !is_direct_constructible<T, double>::value &&
                              !is_direct_constructible<T, int>::value) ||
                             (uncommon_type<T>::value && type_count<T>::value >= 2))>::type> {
  static constexpr object_category value{object_category::tuple_value};
  // the condition on this class requires it be like a tuple, but on some compilers (like Xcode) tuples can be
  // constructed from just the first element so tuples of <string, int,int> can be constructed from a string, which
  // could lead to issues so there are two variants of the condition, the first isolates things with a type size >=2
  // mainly to get tuples on Xcode with the exception of wrappers, the second is the main one and just separating out
  // those cases that are caught by other object classifications
};

/// container type
template <typename T>
struct classify_object<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
  static constexpr object_category value{object_category::container_value};
};

// Type name print

/// Was going to be based on
///  http://stackoverflow.com/questions/1055452/c-get-name-of-type-in-template
/// But this is cleaner and works better in this case

template <typename T,
          enable_if_t<classify_object<T>::value == object_category::char_value, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "CHAR";
}

template <typename T,
          enable_if_t<classify_object<T>::value == object_category::integral_value ||
                          classify_object<T>::value == object_category::integer_constructible,
                      detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "INT";
}

template <typename T,
          enable_if_t<classify_object<T>::value == object_category::unsigned_integral, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "UINT";
}

template <typename T,
          enable_if_t<classify_object<T>::value == object_category::floating_point ||
                          classify_object<T>::value == object_category::number_constructible ||
                          classify_object<T>::value == object_category::double_constructible,
                      detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "FLOAT";
}

/// Print name for enumeration types
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::enumeration, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "ENUM";
}

/// Print name for enumeration types
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::boolean_value, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "BOOLEAN";
}

/// Print name for enumeration types
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::complex_number, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "COMPLEX";
}

/// Print for all other types
template <typename T,
          enable_if_t<classify_object<T>::value >= object_category::string_assignable &&
                          classify_object<T>::value <= object_category::other,
                      detail::enabler> = detail::dummy>
constexpr const char *type_name() {
  return "TEXT";
}
/// typename for tuple value
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count_base<T>::value >= 2,
                      detail::enabler> = detail::dummy>
std::string type_name(); // forward declaration

/// Generate type name for a wrapper or container value
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::container_value ||
                          classify_object<T>::value == object_category::wrapper_value,
                      detail::enabler> = detail::dummy>
std::string type_name(); // forward declaration

/// Print name for single element tuple types
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count_base<T>::value == 1,
                      detail::enabler> = detail::dummy>
inline std::string type_name() {
  return type_name<typename std::decay<typename std::tuple_element<0, T>::type>::type>();
}

/// Empty string if the index > tuple size
template <typename T, std::size_t I>
inline typename std::enable_if<I == type_count_base<T>::value, std::string>::type tuple_name() {
  return std::string{};
}

/// Recursively generate the tuple type name
template <typename T, std::size_t I>
inline typename std::enable_if<(I < type_count_base<T>::value), std::string>::type tuple_name() {
  auto str = std::string{type_name<typename std::decay<typename std::tuple_element<I, T>::type>::type>()} + ',' +
             tuple_name<T, I + 1>();
  if (str.back() == ',')
    str.pop_back();
  return str;
}

/// Print type name for tuples with 2 or more elements
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count_base<T>::value >= 2,
                      detail::enabler>>
inline std::string type_name() {
  auto tname = std::string(1, '[') + tuple_name<T, 0>();
  tname.push_back(']');
  return tname;
}

/// get the type name for a type that has a value_type member
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::container_value ||
                          classify_object<T>::value == object_category::wrapper_value,
                      detail::enabler>>
inline std::string type_name() {
  return type_name<typename T::value_type>();
}

// Lexical cast

/// Convert to an unsigned integral
template <typename T, enable_if_t<std::is_unsigned<T>::value, detail::enabler> = detail::dummy>
bool integral_conversion(const std::string &input, T &output) noexcept {
  if (input.empty() || input.front() == '-') {
    return false;
  }
  char *val{nullptr};
  errno = 0;
  std::uint64_t output_ll = std::strtoull(input.c_str(), &val, 0);
  if (errno == ERANGE) {
    return false;
  }
  output = static_cast<T>(output_ll);
  if (val == (input.c_str() + input.size()) && static_cast<std::uint64_t>(output) == output_ll) {
    return true;
  }
  val = nullptr;
  std::int64_t output_sll = std::strtoll(input.c_str(), &val, 0);
  if (val == (input.c_str() + input.size())) {
    output = (output_sll < 0) ? static_cast<T>(0) : static_cast<T>(output_sll);
    return (static_cast<std::int64_t>(output) == output_sll);
  }
  // remove separators
  if (input.find_first_of("_'") != std::string::npos) {
    std::string nstring = input;
    nstring.erase(std::remove(nstring.begin(), nstring.end(), '_'), nstring.end());
    nstring.erase(std::remove(nstring.begin(), nstring.end(), '\''), nstring.end());
    return integral_conversion(nstring, output);
  }
  if (input.compare(0, 2, "0o") == 0) {
    val = nullptr;
    errno = 0;
    output_ll = std::strtoull(input.c_str() + 2, &val, 8);
    if (errno == ERANGE) {
      return false;
    }
    output = static_cast<T>(output_ll);
    return (val == (input.c_str() + input.size()) && static_cast<std::uint64_t>(output) == output_ll);
  }
  if (input.compare(0, 2, "0b") == 0) {
    val = nullptr;
    errno = 0;
    output_ll = std::strtoull(input.c_str() + 2, &val, 2);
    if (errno == ERANGE) {
      return false;
    }
    output = static_cast<T>(output_ll);
    return (val == (input.c_str() + input.size()) && static_cast<std::uint64_t>(output) == output_ll);
  }
  return false;
}

/// Convert to a signed integral
template <typename T, enable_if_t<std::is_signed<T>::value, detail::enabler> = detail::dummy>
bool integral_conversion(const std::string &input, T &output) noexcept {
  if (input.empty()) {
    return false;
  }
  char *val = nullptr;
  errno = 0;
  std::int64_t output_ll = std::strtoll(input.c_str(), &val, 0);
  if (errno == ERANGE) {
    return false;
  }
  output = static_cast<T>(output_ll);
  if (val == (input.c_str() + input.size()) && static_cast<std::int64_t>(output) == output_ll) {
    return true;
  }
  if (input == "true") {
    // this is to deal with a few oddities with flags and wrapper int types
    output = static_cast<T>(1);
    return true;
  }
  // remove separators
  if (input.find_first_of("_'") != std::string::npos) {
    std::string nstring = input;
    nstring.erase(std::remove(nstring.begin(), nstring.end(), '_'), nstring.end());
    nstring.erase(std::remove(nstring.begin(), nstring.end(), '\''), nstring.end());
    return integral_conversion(nstring, output);
  }
  if (input.compare(0, 2, "0o") == 0) {
    val = nullptr;
    errno = 0;
    output_ll = std::strtoll(input.c_str() + 2, &val, 8);
    if (errno == ERANGE) {
      return false;
    }
    output = static_cast<T>(output_ll);
    return (val == (input.c_str() + input.size()) && static_cast<std::int64_t>(output) == output_ll);
  }
  if (input.compare(0, 2, "0b") == 0) {
    val = nullptr;
    errno = 0;
    output_ll = std::strtoll(input.c_str() + 2, &val, 2);
    if (errno == ERANGE) {
      return false;
    }
    output = static_cast<T>(output_ll);
    return (val == (input.c_str() + input.size()) && static_cast<std::int64_t>(output) == output_ll);
  }
  return false;
}

/// Convert a flag into an integer value  typically binary flags sets errno to nonzero if conversion failed
inline std::int64_t to_flag_value(std::string val) noexcept {
  static const std::string trueString("true");
  static const std::string falseString("false");
  if (val == trueString) {
    return 1;
  }
  if (val == falseString) {
    return -1;
  }
  val = detail::to_lower(val);
  std::int64_t ret = 0;
  if (val.size() == 1) {
    if (val[0] >= '1' && val[0] <= '9') {
      return (static_cast<std::int64_t>(val[0]) - '0');
    }
    switch (val[0]) {
    case '0':
    case 'f':
    case 'n':
    case '-':
      ret = -1;
      break;
    case 't':
    case 'y':
    case '+':
      ret = 1;
      break;
    default:
      errno = EINVAL;
      return -1;
    }
    return ret;
  }
  if (val == trueString || val == "on" || val == "yes" || val == "enable") {
    ret = 1;
  } else if (val == falseString || val == "off" || val == "no" || val == "disable") {
    ret = -1;
  } else {
    char *loc_ptr{nullptr};
    ret = std::strtoll(val.c_str(), &loc_ptr, 0);
    if (loc_ptr != (val.c_str() + val.size()) && errno == 0) {
      errno = EINVAL;
    }
  }
  return ret;
}

/// Integer conversion
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::integral_value ||
                          classify_object<T>::value == object_category::unsigned_integral,
                      detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  return integral_conversion(input, output);
}

/// char values
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::char_value, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  if (input.size() == 1) {
    output = static_cast<T>(input[0]);
    return true;
  }
  return integral_conversion(input, output);
}

/// Boolean values
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::boolean_value, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  errno = 0;
  auto out = to_flag_value(input);
  if (errno == 0) {
    output = (out > 0);
  } else if (errno == ERANGE) {
    output = (input[0] != '-');
  } else {
    return false;
  }
  return true;
}

/// Floats
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::floating_point, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  if (input.empty()) {
    return false;
  }
  char *val = nullptr;
  auto output_ld = std::strtold(input.c_str(), &val);
  output = static_cast<T>(output_ld);
  if (val == (input.c_str() + input.size())) {
    return true;
  }
  // remove separators
  if (input.find_first_of("_'") != std::string::npos) {
    std::string nstring = input;
    nstring.erase(std::remove(nstring.begin(), nstring.end(), '_'), nstring.end());
    nstring.erase(std::remove(nstring.begin(), nstring.end(), '\''), nstring.end());
    return lexical_cast(nstring, output);
  }
  return false;
}

/// complex
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::complex_number, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  using XC = typename wrapped_type<T, double>::type;
  XC x{0.0}, y{0.0};
  auto str1 = input;
  bool worked = false;
  auto nloc = str1.find_last_of("+-");
  if (nloc != std::string::npos && nloc > 0) {
    worked = lexical_cast(str1.substr(0, nloc), x);
    str1 = str1.substr(nloc);
    if (str1.back() == 'i' || str1.back() == 'j')
      str1.pop_back();
    worked = worked && lexical_cast(str1, y);
  } else {
    if (str1.back() == 'i' || str1.back() == 'j') {
      str1.pop_back();
      worked = lexical_cast(str1, y);
      x = XC{0};
    } else {
      worked = lexical_cast(str1, x);
      y = XC{0};
    }
  }
  if (worked) {
    output = T{x, y};
    return worked;
  }
  return from_stream(input, output);
}

/// String and similar direct assignment
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::string_assignable, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  output = input;
  return true;
}

/// String and similar constructible and copy assignment
template <
    typename T,
    enable_if_t<classify_object<T>::value == object_category::string_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  output = T(input);
  return true;
}

/// Wide strings
template <
    typename T,
    enable_if_t<classify_object<T>::value == object_category::wstring_assignable, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  output = widen(input);
  return true;
}

template <
    typename T,
    enable_if_t<classify_object<T>::value == object_category::wstring_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  output = T{widen(input)};
  return true;
}

/// Enumerations
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::enumeration, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  typename std::underlying_type<T>::type val;
  if (!integral_conversion(input, val)) {
    return false;
  }
  output = static_cast<T>(val);
  return true;
}

/// wrapper types
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::wrapper_value &&
                          std::is_assignable<T &, typename T::value_type>::value,
                      detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  typename T::value_type val;
  if (lexical_cast(input, val)) {
    output = val;
    return true;
  }
  return from_stream(input, output);
}

template <typename T,
          enable_if_t<classify_object<T>::value == object_category::wrapper_value &&
                          !std::is_assignable<T &, typename T::value_type>::value && std::is_assignable<T &, T>::value,
                      detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  typename T::value_type val;
  if (lexical_cast(input, val)) {
    output = T{val};
    return true;
  }
  return from_stream(input, output);
}

/// Assignable from double or int
template <
    typename T,
    enable_if_t<classify_object<T>::value == object_category::number_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  int val = 0;
  if (integral_conversion(input, val)) {
    output = T(val);
    return true;
  }

  double dval = 0.0;
  if (lexical_cast(input, dval)) {
    output = T{dval};
    return true;
  }

  return from_stream(input, output);
}

/// Assignable from int
template <
    typename T,
    enable_if_t<classify_object<T>::value == object_category::integer_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  int val = 0;
  if (integral_conversion(input, val)) {
    output = T(val);
    return true;
  }
  return from_stream(input, output);
}

/// Assignable from double
template <
    typename T,
    enable_if_t<classify_object<T>::value == object_category::double_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  double val = 0.0;
  if (lexical_cast(input, val)) {
    output = T{val};
    return true;
  }
  return from_stream(input, output);
}

/// Non-string convertible from an int
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::other && std::is_assignable<T &, int>::value,
                      detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  int val = 0;
  if (integral_conversion(input, val)) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif
    // with Atomic<XX> this could produce a warning due to the conversion but if atomic gets here it is an old style
    // so will most likely still work
    output = val;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    return true;
  }
  // LCOV_EXCL_START
  // This version of cast is only used for odd cases in an older compilers the fail over
  // from_stream is tested elsewhere an not relevant for coverage here
  return from_stream(input, output);
  // LCOV_EXCL_STOP
}

/// Non-string parsable by a stream
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::other && !std::is_assignable<T &, int>::value &&
                          is_istreamable<T>::value,
                      detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
  return from_stream(input, output);
}

/// Fallback overload that prints a human-readable error for types that we don't recognize and that don't have a
/// user-supplied lexical_cast overload.
template <typename T,
          enable_if_t<classify_object<T>::value == object_category::other && !std::is_assignable<T &, int>::value &&
                          !is_istreamable<T>::value && !adl_detail::is_lexical_castable<T>::value,
                      detail::enabler> = detail::dummy>
bool lexical_cast(const std::string & /*input*/, T & /*output*/) {
  static_assert(!std::is_same<T, T>::value, // Can't just write false here.
                "option object type must have a lexical cast overload or streaming input operator(>>) defined, if it "
                "is convertible from another type use the add_option<T, XC>(...) with XC being the known type");
  return false;
}

/// Assign a value through lexical cast operations
/// Strings can be empty so we need to do a little different
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<std::is_same<AssignTo, ConvertTo>::value &&
                          (classify_object<AssignTo>::value == object_category::string_assignable ||
                           classify_object<AssignTo>::value == object_category::string_constructible ||
                           classify_object<AssignTo>::value == object_category::wstring_assignable ||
                           classify_object<AssignTo>::value == object_category::wstring_constructible),
                      detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
  return lexical_cast(input, output);
}

/// Assign a value through lexical cast operations
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<std::is_same<AssignTo, ConvertTo>::value && std::is_assignable<AssignTo &, AssignTo>::value &&
                          classify_object<AssignTo>::value != object_category::string_assignable &&
                          classify_object<AssignTo>::value != object_category::string_constructible &&
                          classify_object<AssignTo>::value != object_category::wstring_assignable &&
                          classify_object<AssignTo>::value != object_category::wstring_constructible,
                      detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
  if (input.empty()) {
    output = AssignTo{};
    return true;
  }

  return lexical_cast(input, output);
}

/// Assign a value through lexical cast operations
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<std::is_same<AssignTo, ConvertTo>::value && !std::is_assignable<AssignTo &, AssignTo>::value &&
                          classify_object<AssignTo>::value == object_category::wrapper_value,
                      detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
  if (input.empty()) {
    typename AssignTo::value_type emptyVal{};
    output = emptyVal;
    return true;
  }
  return lexical_cast(input, output);
}

/// Assign a value through lexical cast operations for int compatible values
/// mainly for atomic operations on some compilers
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<std::is_same<AssignTo, ConvertTo>::value && !std::is_assignable<AssignTo &, AssignTo>::value &&
                          classify_object<AssignTo>::value != object_category::wrapper_value &&
                          std::is_assignable<AssignTo &, int>::value,
                      detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
  if (input.empty()) {
    output = 0;
    return true;
  }
  int val{0};
  if (lexical_cast(input, val)) {
#if defined(__clang__)
/* on some older clang compilers */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#endif
    output = val;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return true;
  }
  return false;
}

/// Assign a value converted from a string in lexical cast to the output value directly
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<!std::is_same<AssignTo, ConvertTo>::value && std::is_assignable<AssignTo &, ConvertTo &>::value,
                      detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
  ConvertTo val{};
  bool parse_result = (!input.empty()) ? lexical_cast(input, val) : true;
  if (parse_result) {
    output = val;
  }
  return parse_result;
}

/// Assign a value from a lexical cast through constructing a value and move assigning it
template <
    typename AssignTo,
    typename ConvertTo,
    enable_if_t<!std::is_same<AssignTo, ConvertTo>::value && !std::is_assignable<AssignTo &, ConvertTo &>::value &&
                    std::is_move_assignable<AssignTo>::value,
                detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
  ConvertTo val{};
  bool parse_result = input.empty() ? true : lexical_cast(input, val);
  if (parse_result) {
    output = AssignTo(val); // use () form of constructor to allow some implicit conversions
  }
  return parse_result;
}

/// primary lexical conversion operation, 1 string to 1 type of some kind
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<classify_object<ConvertTo>::value <= object_category::other &&
                          classify_object<AssignTo>::value <= object_category::wrapper_value,
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
  return lexical_assign<AssignTo, ConvertTo>(strings[0], output);
}

/// Lexical conversion if there is only one element but the conversion type is for two, then call a two element
/// constructor
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<(type_count<AssignTo>::value <= 2) && expected_count<AssignTo>::value == 1 &&
                          is_tuple_like<ConvertTo>::value && type_count_base<ConvertTo>::value == 2,
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
  // the remove const is to handle pair types coming from a container
  using FirstType = typename std::remove_const<typename std::tuple_element<0, ConvertTo>::type>::type;
  using SecondType = typename std::tuple_element<1, ConvertTo>::type;
  FirstType v1;
  SecondType v2;
  bool retval = lexical_assign<FirstType, FirstType>(strings[0], v1);
  retval = retval && lexical_assign<SecondType, SecondType>((strings.size() > 1) ? strings[1] : std::string{}, v2);
  if (retval) {
    output = AssignTo{v1, v2};
  }
  return retval;
}

/// Lexical conversion of a container types of single elements
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
                          type_count<ConvertTo>::value == 1,
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
  output.erase(output.begin(), output.end());
  if (strings.empty()) {
    return true;
  }
  if (strings.size() == 1 && strings[0] == "{}") {
    return true;
  }
  bool skip_remaining = false;
  if (strings.size() == 2 && strings[0] == "{}" && is_separator(strings[1])) {
    skip_remaining = true;
  }
  for (const auto &elem : strings) {
    typename AssignTo::value_type out;
    bool retval = lexical_assign<typename AssignTo::value_type, typename ConvertTo::value_type>(elem, out);
    if (!retval) {
      return false;
    }
    output.insert(output.end(), std::move(out));
    if (skip_remaining) {
      break;
    }
  }
  return (!output.empty());
}

/// Lexical conversion for complex types
template <class AssignTo, class ConvertTo, enable_if_t<is_complex<ConvertTo>::value, detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output) {

  if (strings.size() >= 2 && !strings[1].empty()) {
    using XC2 = typename wrapped_type<ConvertTo, double>::type;
    XC2 x{0.0}, y{0.0};
    auto str1 = strings[1];
    if (str1.back() == 'i' || str1.back() == 'j') {
      str1.pop_back();
    }
    auto worked = lexical_cast(strings[0], x) && lexical_cast(str1, y);
    if (worked) {
      output = ConvertTo{x, y};
    }
    return worked;
  }
  return lexical_assign<AssignTo, ConvertTo>(strings[0], output);
}

/// Conversion to a vector type using a particular single type as the conversion type
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_mutable_container<AssignTo>::value && (expected_count<ConvertTo>::value == 1) &&
                          (type_count<ConvertTo>::value == 1),
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
  bool retval = true;
  output.clear();
  output.reserve(strings.size());
  for (const auto &elem : strings) {

    output.emplace_back();
    retval = retval && lexical_assign<typename AssignTo::value_type, ConvertTo>(elem, output.back());
  }
  return (!output.empty()) && retval;
}

// forward declaration

/// Lexical conversion of a container types with conversion type of two elements
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
                          type_count_base<ConvertTo>::value == 2,
                      detail::enabler> = detail::dummy>
bool lexical_conversion(std::vector<std::string> strings, AssignTo &output);

/// Lexical conversion of a vector types with type_size >2 forward declaration
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
                          type_count_base<ConvertTo>::value != 2 &&
                          ((type_count<ConvertTo>::value > 2) ||
                           (type_count<ConvertTo>::value > type_count_base<ConvertTo>::value)),
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output);

/// Conversion for tuples
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_tuple_like<AssignTo>::value && is_tuple_like<ConvertTo>::value &&
                          (type_count_base<ConvertTo>::value != type_count<ConvertTo>::value ||
                           type_count<ConvertTo>::value > 2),
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output); // forward declaration

/// Conversion for operations where the assigned type is some class but the conversion is a mutable container or large
/// tuple
template <typename AssignTo,
          typename ConvertTo,
          enable_if_t<!is_tuple_like<AssignTo>::value && !is_mutable_container<AssignTo>::value &&
                          classify_object<ConvertTo>::value != object_category::wrapper_value &&
                          (is_mutable_container<ConvertTo>::value || type_count<ConvertTo>::value > 2),
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {

  if (strings.size() > 1 || (!strings.empty() && !(strings.front().empty()))) {
    ConvertTo val;
    auto retval = lexical_conversion<ConvertTo, ConvertTo>(strings, val);
    output = AssignTo{val};
    return retval;
  }
  output = AssignTo{};
  return true;
}

/// function template for converting tuples if the static Index is greater than the tuple size
template <class AssignTo, class ConvertTo, std::size_t I>
inline typename std::enable_if<(I >= type_count_base<AssignTo>::value), bool>::type
tuple_conversion(const std::vector<std::string> &, AssignTo &) {
  return true;
}

/// Conversion of a tuple element where the type size ==1 and not a mutable container
template <class AssignTo, class ConvertTo>
inline typename std::enable_if<!is_mutable_container<ConvertTo>::value && type_count<ConvertTo>::value == 1, bool>::type
tuple_type_conversion(std::vector<std::string> &strings, AssignTo &output) {
  auto retval = lexical_assign<AssignTo, ConvertTo>(strings[0], output);
  strings.erase(strings.begin());
  return retval;
}

/// Conversion of a tuple element where the type size !=1 but the size is fixed and not a mutable container
template <class AssignTo, class ConvertTo>
inline typename std::enable_if<!is_mutable_container<ConvertTo>::value && (type_count<ConvertTo>::value > 1) &&
                                   type_count<ConvertTo>::value == type_count_min<ConvertTo>::value,
                               bool>::type
tuple_type_conversion(std::vector<std::string> &strings, AssignTo &output) {
  auto retval = lexical_conversion<AssignTo, ConvertTo>(strings, output);
  strings.erase(strings.begin(), strings.begin() + type_count<ConvertTo>::value);
  return retval;
}

/// Conversion of a tuple element where the type is a mutable container or a type with different min and max type sizes
template <class AssignTo, class ConvertTo>
inline typename std::enable_if<is_mutable_container<ConvertTo>::value ||
                                   type_count<ConvertTo>::value != type_count_min<ConvertTo>::value,
                               bool>::type
tuple_type_conversion(std::vector<std::string> &strings, AssignTo &output) {

  std::size_t index{subtype_count_min<ConvertTo>::value};
  const std::size_t mx_count{subtype_count<ConvertTo>::value};
  const std::size_t mx{(std::min)(mx_count, strings.size() - 1)};

  while (index < mx) {
    if (is_separator(strings[index])) {
      break;
    }
    ++index;
  }
  bool retval = lexical_conversion<AssignTo, ConvertTo>(
      std::vector<std::string>(strings.begin(), strings.begin() + static_cast<std::ptrdiff_t>(index)), output);
  if (strings.size() > index) {
    strings.erase(strings.begin(), strings.begin() + static_cast<std::ptrdiff_t>(index) + 1);
  } else {
    strings.clear();
  }
  return retval;
}

/// Tuple conversion operation
template <class AssignTo, class ConvertTo, std::size_t I>
inline typename std::enable_if<(I < type_count_base<AssignTo>::value), bool>::type
tuple_conversion(std::vector<std::string> strings, AssignTo &output) {
  bool retval = true;
  using ConvertToElement = typename std::
      conditional<is_tuple_like<ConvertTo>::value, typename std::tuple_element<I, ConvertTo>::type, ConvertTo>::type;
  if (!strings.empty()) {
    retval = retval && tuple_type_conversion<typename std::tuple_element<I, AssignTo>::type, ConvertToElement>(
                           strings, std::get<I>(output));
  }
  retval = retval && tuple_conversion<AssignTo, ConvertTo, I + 1>(std::move(strings), output);
  return retval;
}

/// Lexical conversion of a container types with tuple elements of size 2
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
                          type_count_base<ConvertTo>::value == 2,
                      detail::enabler>>
bool lexical_conversion(std::vector<std::string> strings, AssignTo &output) {
  output.clear();
  while (!strings.empty()) {

    typename std::remove_const<typename std::tuple_element<0, typename ConvertTo::value_type>::type>::type v1;
    typename std::tuple_element<1, typename ConvertTo::value_type>::type v2;
    bool retval = tuple_type_conversion<decltype(v1), decltype(v1)>(strings, v1);
    if (!strings.empty()) {
      retval = retval && tuple_type_conversion<decltype(v2), decltype(v2)>(strings, v2);
    }
    if (retval) {
      output.insert(output.end(), typename AssignTo::value_type{v1, v2});
    } else {
      return false;
    }
  }
  return (!output.empty());
}

/// lexical conversion of tuples with type count>2 or tuples of types of some element with a type size>=2
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_tuple_like<AssignTo>::value && is_tuple_like<ConvertTo>::value &&
                          (type_count_base<ConvertTo>::value != type_count<ConvertTo>::value ||
                           type_count<ConvertTo>::value > 2),
                      detail::enabler>>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
  static_assert(
      !is_tuple_like<ConvertTo>::value || type_count_base<AssignTo>::value == type_count_base<ConvertTo>::value,
      "if the conversion type is defined as a tuple it must be the same size as the type you are converting to");
  return tuple_conversion<AssignTo, ConvertTo, 0>(strings, output);
}

/// Lexical conversion of a vector types for everything but tuples of two elements and types of size 1
template <class AssignTo,
          class ConvertTo,
          enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
                          type_count_base<ConvertTo>::value != 2 &&
                          ((type_count<ConvertTo>::value > 2) ||
                           (type_count<ConvertTo>::value > type_count_base<ConvertTo>::value)),
                      detail::enabler>>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
  bool retval = true;
  output.clear();
  std::vector<std::string> temp;
  std::size_t ii{0};
  std::size_t icount{0};
  std::size_t xcm{type_count<ConvertTo>::value};
  auto ii_max = strings.size();
  while (ii < ii_max) {
    temp.push_back(strings[ii]);
    ++ii;
    ++icount;
    if (icount == xcm || is_separator(temp.back()) || ii == ii_max) {
      if (static_cast<int>(xcm) > type_count_min<ConvertTo>::value && is_separator(temp.back())) {
        temp.pop_back();
      }
      typename AssignTo::value_type temp_out;
      retval = retval &&
               lexical_conversion<typename AssignTo::value_type, typename ConvertTo::value_type>(temp, temp_out);
      temp.clear();
      if (!retval) {
        return false;
      }
      output.insert(output.end(), std::move(temp_out));
      icount = 0;
    }
  }
  return retval;
}

/// conversion for wrapper types
template <typename AssignTo,
          class ConvertTo,
          enable_if_t<classify_object<ConvertTo>::value == object_category::wrapper_value &&
                          std::is_assignable<ConvertTo &, ConvertTo>::value,
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output) {
  if (strings.empty() || strings.front().empty()) {
    output = ConvertTo{};
    return true;
  }
  typename ConvertTo::value_type val;
  if (lexical_conversion<typename ConvertTo::value_type, typename ConvertTo::value_type>(strings, val)) {
    output = ConvertTo{val};
    return true;
  }
  return false;
}

/// conversion for wrapper types
template <typename AssignTo,
          class ConvertTo,
          enable_if_t<classify_object<ConvertTo>::value == object_category::wrapper_value &&
                          !std::is_assignable<AssignTo &, ConvertTo>::value,
                      detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output) {
  using ConvertType = typename ConvertTo::value_type;
  if (strings.empty() || strings.front().empty()) {
    output = ConvertType{};
    return true;
  }
  ConvertType val;
  if (lexical_conversion<typename ConvertTo::value_type, typename ConvertTo::value_type>(strings, val)) {
    output = val;
    return true;
  }
  return false;
}

/// Sum a vector of strings
inline std::string sum_string_vector(const std::vector<std::string> &values) {
  double val{0.0};
  bool fail{false};
  std::string output;
  for (const auto &arg : values) {
    double tv{0.0};
    auto comp = lexical_cast(arg, tv);
    if (!comp) {
      errno = 0;
      auto fv = detail::to_flag_value(arg);
      fail = (errno != 0);
      if (fail) {
        break;
      }
      tv = static_cast<double>(fv);
    }
    val += tv;
  }
  if (fail) {
    for (const auto &arg : values) {
      output.append(arg);
    }
  } else {
    std::ostringstream out;
    out.precision(16);
    out << val;
    output = out.str();
  }
  return output;
}

} // namespace detail

namespace detail {

// Returns false if not a short option. Otherwise, sets opt name and rest and returns true
CLI11_INLINE bool split_short(const std::string &current, std::string &name, std::string &rest);

// Returns false if not a long option. Otherwise, sets opt name and other side of = and returns true
CLI11_INLINE bool split_long(const std::string &current, std::string &name, std::string &value);

// Returns false if not a windows style option. Otherwise, sets opt name and value and returns true
CLI11_INLINE bool split_windows_style(const std::string &current, std::string &name, std::string &value);

// Splits a string into multiple long and short names
CLI11_INLINE std::vector<std::string> split_names(std::string current);

/// extract default flag values either {def} or starting with a !
CLI11_INLINE std::vector<std::pair<std::string, std::string>> get_default_flag_values(const std::string &str);

/// Get a vector of short names, one of long names, and a single name
CLI11_INLINE std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>
get_names(const std::vector<std::string> &input);

} // namespace detail

namespace detail {

CLI11_INLINE bool split_short(const std::string &current, std::string &name, std::string &rest) {
  if (current.size() > 1 && current[0] == '-' && valid_first_char(current[1])) {
    name = current.substr(1, 1);
    rest = current.substr(2);
    return true;
  }
  return false;
}

CLI11_INLINE bool split_long(const std::string &current, std::string &name, std::string &value) {
  if (current.size() > 2 && current.compare(0, 2, "--") == 0 && valid_first_char(current[2])) {
    auto loc = current.find_first_of('=');
    if (loc != std::string::npos) {
      name = current.substr(2, loc - 2);
      value = current.substr(loc + 1);
    } else {
      name = current.substr(2);
      value = "";
    }
    return true;
  }
  return false;
}

CLI11_INLINE bool split_windows_style(const std::string &current, std::string &name, std::string &value) {
  if (current.size() > 1 && current[0] == '/' && valid_first_char(current[1])) {
    auto loc = current.find_first_of(':');
    if (loc != std::string::npos) {
      name = current.substr(1, loc - 1);
      value = current.substr(loc + 1);
    } else {
      name = current.substr(1);
      value = "";
    }
    return true;
  }
  return false;
}

CLI11_INLINE std::vector<std::string> split_names(std::string current) {
  std::vector<std::string> output;
  std::size_t val = 0;
  while ((val = current.find(',')) != std::string::npos) {
    output.push_back(trim_copy(current.substr(0, val)));
    current = current.substr(val + 1);
  }
  output.push_back(trim_copy(current));
  return output;
}

CLI11_INLINE std::vector<std::pair<std::string, std::string>> get_default_flag_values(const std::string &str) {
  std::vector<std::string> flags = split_names(str);
  flags.erase(std::remove_if(flags.begin(),
                             flags.end(),
                             [](const std::string &name) {
                               return ((name.empty()) || (!(((name.find_first_of('{') != std::string::npos) &&
                                                             (name.back() == '}')) ||
                                                            (name[0] == '!'))));
                             }),
              flags.end());
  std::vector<std::pair<std::string, std::string>> output;
  output.reserve(flags.size());
  for (auto &flag : flags) {
    auto def_start = flag.find_first_of('{');
    std::string defval = "false";
    if ((def_start != std::string::npos) && (flag.back() == '}')) {
      defval = flag.substr(def_start + 1);
      defval.pop_back();
      flag.erase(def_start, std::string::npos); // NOLINT(readability-suspicious-call-argument)
    }
    flag.erase(0, flag.find_first_not_of("-!"));
    output.emplace_back(flag, defval);
  }
  return output;
}

CLI11_INLINE std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>
get_names(const std::vector<std::string> &input) {

  std::vector<std::string> short_names;
  std::vector<std::string> long_names;
  std::string pos_name;
  for (std::string name : input) {
    if (name.length() == 0) {
      continue;
    }
    if (name.length() > 1 && name[0] == '-' && name[1] != '-') {
      if (name.length() == 2 && valid_first_char(name[1]))
        short_names.emplace_back(1, name[1]);
      else if (name.length() > 2)
        throw BadNameString::MissingDash(name);
      else
        throw BadNameString::OneCharName(name);
    } else if (name.length() > 2 && name.substr(0, 2) == "--") {
      name = name.substr(2);
      if (valid_name_string(name))
        long_names.push_back(name);
      else
        throw BadNameString::BadLongName(name);
    } else if (name == "-" || name == "--") {
      throw BadNameString::DashesOnly(name);
    } else {
      if (!pos_name.empty())
        throw BadNameString::MultiPositionalNames(name);
      if (valid_name_string(name)) {
        pos_name = name;
      } else {
        throw BadNameString::BadPositionalName(name);
      }
    }
  }
  return std::make_tuple(short_names, long_names, pos_name);
}

} // namespace detail

class App;

/// Holds values to load into Options
struct ConfigItem {
  /// This is the list of parents
  std::vector<std::string> parents{};

  /// This is the name
  std::string name{};
  /// Listing of inputs
  std::vector<std::string> inputs{};

  /// The list of parents and name joined by "."
  CLI11_NODISCARD std::string fullname() const {
    std::vector<std::string> tmp = parents;
    tmp.emplace_back(name);
    return detail::join(tmp, ".");
  }
};

/// This class provides a converter for configuration files.
class Config {
protected:
  std::vector<ConfigItem> items{};

public:
  /// Convert an app into a configuration
  virtual std::string to_config(const App *, bool, bool, std::string) const = 0;

  /// Convert a configuration into an app
  virtual std::vector<ConfigItem> from_config(std::istream &) const = 0;

  /// Get a flag value
  CLI11_NODISCARD virtual std::string to_flag(const ConfigItem &item) const {
    if (item.inputs.size() == 1) {
      return item.inputs.at(0);
    }
    if (item.inputs.empty()) {
      return "{}";
    }
    throw ConversionError::TooManyInputsFlag(item.fullname()); // LCOV_EXCL_LINE
  }

  /// Parse a config file, throw an error (ParseError:ConfigParseError or FileError) on failure
  CLI11_NODISCARD std::vector<ConfigItem> from_file(const std::string &name) const {
    std::ifstream input{name};
    if (!input.good())
      throw FileError::Missing(name);

    return from_config(input);
  }

  /// Virtual destructor
  virtual ~Config() = default;
};

/// This converter works with INI/TOML files; to write INI files use ConfigINI
class ConfigBase : public Config {
protected:
  /// the character used for comments
  char commentChar = '#';
  /// the character used to start an array '\0' is a default to not use
  char arrayStart = '[';
  /// the character used to end an array '\0' is a default to not use
  char arrayEnd = ']';
  /// the character used to separate elements in an array
  char arraySeparator = ',';
  /// the character used separate the name from the value
  char valueDelimiter = '=';
  /// the character to use around strings
  char stringQuote = '"';
  /// the character to use around single characters and literal strings
  char literalQuote = '\'';
  /// the maximum number of layers to allow
  uint8_t maximumLayers{255};
  /// the separator used to separator parent layers
  char parentSeparatorChar{'.'};
  /// Specify the configuration index to use for arrayed sections
  int16_t configIndex{-1};
  /// Specify the configuration section that should be used
  std::string configSection{};

public:
  std::string
  to_config(const App * /*app*/, bool default_also, bool write_description, std::string prefix) const override;

  std::vector<ConfigItem> from_config(std::istream &input) const override;
  /// Specify the configuration for comment characters
  ConfigBase *comment(char cchar) {
    commentChar = cchar;
    return this;
  }
  /// Specify the start and end characters for an array
  ConfigBase *arrayBounds(char aStart, char aEnd) {
    arrayStart = aStart;
    arrayEnd = aEnd;
    return this;
  }
  /// Specify the delimiter character for an array
  ConfigBase *arrayDelimiter(char aSep) {
    arraySeparator = aSep;
    return this;
  }
  /// Specify the delimiter between a name and value
  ConfigBase *valueSeparator(char vSep) {
    valueDelimiter = vSep;
    return this;
  }
  /// Specify the quote characters used around strings and literal strings
  ConfigBase *quoteCharacter(char qString, char literalChar) {
    stringQuote = qString;
    literalQuote = literalChar;
    return this;
  }
  /// Specify the maximum number of parents
  ConfigBase *maxLayers(uint8_t layers) {
    maximumLayers = layers;
    return this;
  }
  /// Specify the separator to use for parent layers
  ConfigBase *parentSeparator(char sep) {
    parentSeparatorChar = sep;
    return this;
  }
  /// get a reference to the configuration section
  std::string &sectionRef() { return configSection; }
  /// get the section
  CLI11_NODISCARD const std::string &section() const { return configSection; }
  /// specify a particular section of the configuration file to use
  ConfigBase *section(const std::string &sectionName) {
    configSection = sectionName;
    return this;
  }

  /// get a reference to the configuration index
  int16_t &indexRef() { return configIndex; }
  /// get the section index
  CLI11_NODISCARD int16_t index() const { return configIndex; }
  /// specify a particular index in the section to use (-1) for all sections to use
  ConfigBase *index(int16_t sectionIndex) {
    configIndex = sectionIndex;
    return this;
  }
};

/// the default Config is the TOML file format
using ConfigTOML = ConfigBase;

/// ConfigINI generates a "standard" INI compliant output
class ConfigINI : public ConfigTOML {

public:
  ConfigINI() {
    commentChar = ';';
    arrayStart = '\0';
    arrayEnd = '\0';
    arraySeparator = ' ';
    valueDelimiter = '=';
  }
};

class Option;

/// @defgroup validator_group Validators

/// @brief Some validators that are provided
///
/// These are simple `std::string(const std::string&)` validators that are useful. They return
/// a string if the validation fails. A custom struct is provided, as well, with the same user
/// semantics, but with the ability to provide a new type name.
/// @{

///
class Validator {
protected:
  /// This is the description function, if empty the description_ will be used
  std::function<std::string()> desc_function_{[]() { return std::string{}; }};

  /// This is the base function that is to be called.
  /// Returns a string error message if validation fails.
  std::function<std::string(std::string &)> func_{[](std::string &) { return std::string{}; }};
  /// The name for search purposes of the Validator
  std::string name_{};
  /// A Validator will only apply to an indexed value (-1 is all elements)
  int application_index_ = -1;
  /// Enable for Validator to allow it to be disabled if need be
  bool active_{true};
  /// specify that a validator should not modify the input
  bool non_modifying_{false};

  Validator(std::string validator_desc, std::function<std::string(std::string &)> func)
      : desc_function_([validator_desc]() { return validator_desc; }), func_(std::move(func)) {}

public:
  Validator() = default;
  /// Construct a Validator with just the description string
  explicit Validator(std::string validator_desc) : desc_function_([validator_desc]() { return validator_desc; }) {}
  /// Construct Validator from basic information
  Validator(std::function<std::string(std::string &)> op, std::string validator_desc, std::string validator_name = "")
      : desc_function_([validator_desc]() { return validator_desc; }), func_(std::move(op)),
        name_(std::move(validator_name)) {}
  /// Set the Validator operation function
  Validator &operation(std::function<std::string(std::string &)> op) {
    func_ = std::move(op);
    return *this;
  }
  /// This is the required operator for a Validator - provided to help
  /// users (CLI11 uses the member `func` directly)
  std::string operator()(std::string &str) const;

  /// This is the required operator for a Validator - provided to help
  /// users (CLI11 uses the member `func` directly)
  std::string operator()(const std::string &str) const {
    std::string value = str;
    return (active_) ? func_(value) : std::string{};
  }

  /// Specify the type string
  Validator &description(std::string validator_desc) {
    desc_function_ = [validator_desc]() { return validator_desc; };
    return *this;
  }
  /// Specify the type string
  CLI11_NODISCARD Validator description(std::string validator_desc) const;

  /// Generate type description information for the Validator
  CLI11_NODISCARD std::string get_description() const {
    if (active_) {
      return desc_function_();
    }
    return std::string{};
  }
  /// Specify the type string
  Validator &name(std::string validator_name) {
    name_ = std::move(validator_name);
    return *this;
  }
  /// Specify the type string
  CLI11_NODISCARD Validator name(std::string validator_name) const {
    Validator newval(*this);
    newval.name_ = std::move(validator_name);
    return newval;
  }
  /// Get the name of the Validator
  CLI11_NODISCARD const std::string &get_name() const { return name_; }
  /// Specify whether the Validator is active or not
  Validator &active(bool active_val = true) {
    active_ = active_val;
    return *this;
  }
  /// Specify whether the Validator is active or not
  CLI11_NODISCARD Validator active(bool active_val = true) const {
    Validator newval(*this);
    newval.active_ = active_val;
    return newval;
  }

  /// Specify whether the Validator can be modifying or not
  Validator &non_modifying(bool no_modify = true) {
    non_modifying_ = no_modify;
    return *this;
  }
  /// Specify the application index of a validator
  Validator &application_index(int app_index) {
    application_index_ = app_index;
    return *this;
  }
  /// Specify the application index of a validator
  CLI11_NODISCARD Validator application_index(int app_index) const {
    Validator newval(*this);
    newval.application_index_ = app_index;
    return newval;
  }
  /// Get the current value of the application index
  CLI11_NODISCARD int get_application_index() const { return application_index_; }
  /// Get a boolean if the validator is active
  CLI11_NODISCARD bool get_active() const { return active_; }

  /// Get a boolean if the validator is allowed to modify the input returns true if it can modify the input
  CLI11_NODISCARD bool get_modifying() const { return !non_modifying_; }

  /// Combining validators is a new validator. Type comes from left validator if function, otherwise only set if the
  /// same.
  Validator operator&(const Validator &other) const;

  /// Combining validators is a new validator. Type comes from left validator if function, otherwise only set if the
  /// same.
  Validator operator|(const Validator &other) const;

  /// Create a validator that fails when a given validator succeeds
  Validator operator!() const;

private:
  void _merge_description(const Validator &val1, const Validator &val2, const std::string &merger);
};

/// Class wrapping some of the accessors of Validator
class CustomValidator : public Validator {
public:
};
// The implementation of the built in validators is using the Validator class;
// the user is only expected to use the const (static) versions (since there's no setup).
// Therefore, this is in detail.
namespace detail {

/// CLI enumeration of different file types
enum class path_type { nonexistent,
                       file,
                       directory };

/// get the type of the path from a file name
CLI11_INLINE path_type check_path(const char *file) noexcept;

/// Check for an existing file (returns error message if check fails)
class ExistingFileValidator : public Validator {
public:
  ExistingFileValidator();
};

/// Check for an existing directory (returns error message if check fails)
class ExistingDirectoryValidator : public Validator {
public:
  ExistingDirectoryValidator();
};

/// Check for an existing path
class ExistingPathValidator : public Validator {
public:
  ExistingPathValidator();
};

/// Check for an non-existing path
class NonexistentPathValidator : public Validator {
public:
  NonexistentPathValidator();
};

/// Validate the given string is a legal ipv4 address
class IPV4Validator : public Validator {
public:
  IPV4Validator();
};

class EscapedStringTransformer : public Validator {
public:
  EscapedStringTransformer();
};

} // namespace detail

// Static is not needed here, because global const implies static.

/// Check for existing file (returns error message if check fails)
const detail::ExistingFileValidator ExistingFile;

/// Check for an existing directory (returns error message if check fails)
const detail::ExistingDirectoryValidator ExistingDirectory;

/// Check for an existing path
const detail::ExistingPathValidator ExistingPath;

/// Check for an non-existing path
const detail::NonexistentPathValidator NonexistentPath;

/// Check for an IP4 address
const detail::IPV4Validator ValidIPV4;

/// convert escaped characters into their associated values
const detail::EscapedStringTransformer EscapedString;

/// Validate the input as a particular type
template <typename DesiredType>
class TypeValidator : public Validator {
public:
  explicit TypeValidator(const std::string &validator_name)
      : Validator(validator_name, [](std::string &input_string) {
          using CLI::detail::lexical_cast;
          auto val = DesiredType();
          if (!lexical_cast(input_string, val)) {
            return std::string("Failed parsing ") + input_string + " as a " + detail::type_name<DesiredType>();
          }
          return std::string();
        }) {}
  TypeValidator() : TypeValidator(detail::type_name<DesiredType>()) {}
};

/// Check for a number
const TypeValidator<double> Number("NUMBER");

/// Modify a path if the file is a particular default location, can be used as Check or transform
/// with the error return optionally disabled
class FileOnDefaultPath : public Validator {
public:
  explicit FileOnDefaultPath(std::string default_path, bool enableErrorReturn = true);
};

/// Produce a range (factory). Min and max are inclusive.
class Range : public Validator {
public:
  /// This produces a range with min and max inclusive.
  ///
  /// Note that the constructor is templated, but the struct is not, so C++17 is not
  /// needed to provide nice syntax for Range(a,b).
  template <typename T>
  Range(T min_val, T max_val, const std::string &validator_name = std::string{}) : Validator(validator_name) {
    if (validator_name.empty()) {
      std::stringstream out;
      out << detail::type_name<T>() << " in [" << min_val << " - " << max_val << "]";
      description(out.str());
    }

    func_ = [min_val, max_val](std::string &input) {
      using CLI::detail::lexical_cast;
      T val;
      bool converted = lexical_cast(input, val);
      if ((!converted) || (val < min_val || val > max_val)) {
        std::stringstream out;
        out << "Value " << input << " not in range [";
        out << min_val << " - " << max_val << "]";
        return out.str();
      }
      return std::string{};
    };
  }

  /// Range of one value is 0 to value
  template <typename T>
  explicit Range(T max_val, const std::string &validator_name = std::string{})
      : Range(static_cast<T>(0), max_val, validator_name) {}
};

/// Check for a non negative number
const Range NonNegativeNumber((std::numeric_limits<double>::max)(), "NONNEGATIVE");

/// Check for a positive valued number (val>0.0), <double>::min  here is the smallest positive number
const Range PositiveNumber((std::numeric_limits<double>::min)(), (std::numeric_limits<double>::max)(), "POSITIVE");

/// Produce a bounded range (factory). Min and max are inclusive.
class Bound : public Validator {
public:
  /// This bounds a value with min and max inclusive.
  ///
  /// Note that the constructor is templated, but the struct is not, so C++17 is not
  /// needed to provide nice syntax for Range(a,b).
  template <typename T>
  Bound(T min_val, T max_val) {
    std::stringstream out;
    out << detail::type_name<T>() << " bounded to [" << min_val << " - " << max_val << "]";
    description(out.str());

    func_ = [min_val, max_val](std::string &input) {
      using CLI::detail::lexical_cast;
      T val;
      bool converted = lexical_cast(input, val);
      if (!converted) {
        return std::string("Value ") + input + " could not be converted";
      }
      if (val < min_val)
        input = detail::to_string(min_val);
      else if (val > max_val)
        input = detail::to_string(max_val);

      return std::string{};
    };
  }

  /// Range of one value is 0 to value
  template <typename T>
  explicit Bound(T max_val) : Bound(static_cast<T>(0), max_val) {}
};

namespace detail {
template <typename T,
          enable_if_t<is_copyable_ptr<typename std::remove_reference<T>::type>::value, detail::enabler> = detail::dummy>
auto smart_deref(T value) -> decltype(*value) {
  return *value;
}

template <
    typename T,
    enable_if_t<!is_copyable_ptr<typename std::remove_reference<T>::type>::value, detail::enabler> = detail::dummy>
typename std::remove_reference<T>::type &smart_deref(T &value) {
  return value;
}
/// Generate a string representation of a set
template <typename T>
std::string generate_set(const T &set) {
  using element_t = typename detail::element_type<T>::type;
  using iteration_type_t = typename detail::pair_adaptor<element_t>::value_type; // the type of the object pair
  std::string out(1, '{');
  out.append(detail::join(
      detail::smart_deref(set),
      [](const iteration_type_t &v) { return detail::pair_adaptor<element_t>::first(v); },
      ","));
  out.push_back('}');
  return out;
}

/// Generate a string representation of a map
template <typename T>
std::string generate_map(const T &map, bool key_only = false) {
  using element_t = typename detail::element_type<T>::type;
  using iteration_type_t = typename detail::pair_adaptor<element_t>::value_type; // the type of the object pair
  std::string out(1, '{');
  out.append(detail::join(
      detail::smart_deref(map),
      [key_only](const iteration_type_t &v) {
        std::string res{detail::to_string(detail::pair_adaptor<element_t>::first(v))};

        if (!key_only) {
          res.append("->");
          res += detail::to_string(detail::pair_adaptor<element_t>::second(v));
        }
        return res;
      },
      ","));
  out.push_back('}');
  return out;
}

template <typename C, typename V>
struct has_find {
  template <typename CC, typename VV>
  static auto test(int) -> decltype(std::declval<CC>().find(std::declval<VV>()), std::true_type());
  template <typename, typename>
  static auto test(...) -> decltype(std::false_type());

  static const auto value = decltype(test<C, V>(0))::value;
  using type = std::integral_constant<bool, value>;
};

/// A search function
template <typename T, typename V, enable_if_t<!has_find<T, V>::value, detail::enabler> = detail::dummy>
auto search(const T &set, const V &val) -> std::pair<bool, decltype(std::begin(detail::smart_deref(set)))> {
  using element_t = typename detail::element_type<T>::type;
  auto &setref = detail::smart_deref(set);
  auto it = std::find_if(std::begin(setref), std::end(setref), [&val](decltype(*std::begin(setref)) v) {
    return (detail::pair_adaptor<element_t>::first(v) == val);
  });
  return {(it != std::end(setref)), it};
}

/// A search function that uses the built in find function
template <typename T, typename V, enable_if_t<has_find<T, V>::value, detail::enabler> = detail::dummy>
auto search(const T &set, const V &val) -> std::pair<bool, decltype(std::begin(detail::smart_deref(set)))> {
  auto &setref = detail::smart_deref(set);
  auto it = setref.find(val);
  return {(it != std::end(setref)), it};
}

/// A search function with a filter function
template <typename T, typename V>
auto search(const T &set, const V &val, const std::function<V(V)> &filter_function)
    -> std::pair<bool, decltype(std::begin(detail::smart_deref(set)))> {
  using element_t = typename detail::element_type<T>::type;
  // do the potentially faster first search
  auto res = search(set, val);
  if ((res.first) || (!(filter_function))) {
    return res;
  }
  // if we haven't found it do the longer linear search with all the element translations
  auto &setref = detail::smart_deref(set);
  auto it = std::find_if(std::begin(setref), std::end(setref), [&](decltype(*std::begin(setref)) v) {
    V a{detail::pair_adaptor<element_t>::first(v)};
    a = filter_function(a);
    return (a == val);
  });
  return {(it != std::end(setref)), it};
}

// the following suggestion was made by Nikita Ofitserov(@himikof)
// done in templates to prevent compiler warnings on negation of unsigned numbers

/// Do a check for overflow on signed numbers
template <typename T>
inline typename std::enable_if<std::is_signed<T>::value, T>::type overflowCheck(const T &a, const T &b) {
  if ((a > 0) == (b > 0)) {
    return ((std::numeric_limits<T>::max)() / (std::abs)(a) < (std::abs)(b));
  }
  return ((std::numeric_limits<T>::min)() / (std::abs)(a) > -(std::abs)(b));
}
/// Do a check for overflow on unsigned numbers
template <typename T>
inline typename std::enable_if<!std::is_signed<T>::value, T>::type overflowCheck(const T &a, const T &b) {
  return ((std::numeric_limits<T>::max)() / a < b);
}

/// Performs a *= b; if it doesn't cause integer overflow. Returns false otherwise.
template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type checked_multiply(T &a, T b) {
  if (a == 0 || b == 0 || a == 1 || b == 1) {
    a *= b;
    return true;
  }
  if (a == (std::numeric_limits<T>::min)() || b == (std::numeric_limits<T>::min)()) {
    return false;
  }
  if (overflowCheck(a, b)) {
    return false;
  }
  a *= b;
  return true;
}

/// Performs a *= b; if it doesn't equal infinity. Returns false otherwise.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type checked_multiply(T &a, T b) {
  T c = a * b;
  if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) {
    return false;
  }
  a = c;
  return true;
}

} // namespace detail
/// Verify items are in a set
class IsMember : public Validator {
public:
  using filter_fn_t = std::function<std::string(std::string)>;

  /// This allows in-place construction using an initializer list
  template <typename T, typename... Args>
  IsMember(std::initializer_list<T> values, Args &&...args)
      : IsMember(std::vector<T>(values), std::forward<Args>(args)...) {}

  /// This checks to see if an item is in a set (empty function)
  template <typename T>
  explicit IsMember(T &&set) : IsMember(std::forward<T>(set), nullptr) {}

  /// This checks to see if an item is in a set: pointer or copy version. You can pass in a function that will filter
  /// both sides of the comparison before computing the comparison.
  template <typename T, typename F>
  explicit IsMember(T set, F filter_function) {

    // Get the type of the contained item - requires a container have ::value_type
    // if the type does not have first_type and second_type, these are both value_type
    using element_t = typename detail::element_type<T>::type;            // Removes (smart) pointers if needed
    using item_t = typename detail::pair_adaptor<element_t>::first_type; // Is value_type if not a map

    using local_item_t = typename IsMemberType<item_t>::type; // This will convert bad types to good ones
                                                              // (const char * to std::string)

    // Make a local copy of the filter function, using a std::function if not one already
    std::function<local_item_t(local_item_t)> filter_fn = filter_function;

    // This is the type name for help, it will take the current version of the set contents
    desc_function_ = [set]() { return detail::generate_set(detail::smart_deref(set)); };

    // This is the function that validates
    // It stores a copy of the set pointer-like, so shared_ptr will stay alive
    func_ = [set, filter_fn](std::string &input) {
      using CLI::detail::lexical_cast;
      local_item_t b;
      if (!lexical_cast(input, b)) {
        throw ValidationError(input); // name is added later
      }
      if (filter_fn) {
        b = filter_fn(b);
      }
      auto res = detail::search(set, b, filter_fn);
      if (res.first) {
        // Make sure the version in the input string is identical to the one in the set
        if (filter_fn) {
          input = detail::value_string(detail::pair_adaptor<element_t>::first(*(res.second)));
        }

        // Return empty error string (success)
        return std::string{};
      }

      // If you reach this point, the result was not found
      return input + " not in " + detail::generate_set(detail::smart_deref(set));
    };
  }

  /// You can pass in as many filter functions as you like, they nest (string only currently)
  template <typename T, typename... Args>
  IsMember(T &&set, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&...other)
      : IsMember(
            std::forward<T>(set),
            [filter_fn_1, filter_fn_2](std::string a) { return filter_fn_2(filter_fn_1(a)); },
            other...) {}
};

/// definition of the default transformation object
template <typename T>
using TransformPairs = std::vector<std::pair<std::string, T>>;

/// Translate named items to other or a value set
class Transformer : public Validator {
public:
  using filter_fn_t = std::function<std::string(std::string)>;

  /// This allows in-place construction
  template <typename... Args>
  Transformer(std::initializer_list<std::pair<std::string, std::string>> values, Args &&...args)
      : Transformer(TransformPairs<std::string>(values), std::forward<Args>(args)...) {}

  /// direct map of std::string to std::string
  template <typename T>
  explicit Transformer(T &&mapping) : Transformer(std::forward<T>(mapping), nullptr) {}

  /// This checks to see if an item is in a set: pointer or copy version. You can pass in a function that will filter
  /// both sides of the comparison before computing the comparison.
  template <typename T, typename F>
  explicit Transformer(T mapping, F filter_function) {

    static_assert(detail::pair_adaptor<typename detail::element_type<T>::type>::value,
                  "mapping must produce value pairs");
    // Get the type of the contained item - requires a container have ::value_type
    // if the type does not have first_type and second_type, these are both value_type
    using element_t = typename detail::element_type<T>::type;            // Removes (smart) pointers if needed
    using item_t = typename detail::pair_adaptor<element_t>::first_type; // Is value_type if not a map
    using local_item_t = typename IsMemberType<item_t>::type;            // Will convert bad types to good ones
                                                                         // (const char * to std::string)

    // Make a local copy of the filter function, using a std::function if not one already
    std::function<local_item_t(local_item_t)> filter_fn = filter_function;

    // This is the type name for help, it will take the current version of the set contents
    desc_function_ = [mapping]() { return detail::generate_map(detail::smart_deref(mapping)); };

    func_ = [mapping, filter_fn](std::string &input) {
      using CLI::detail::lexical_cast;
      local_item_t b;
      if (!lexical_cast(input, b)) {
        return std::string();
        // there is no possible way we can match anything in the mapping if we can't convert so just return
      }
      if (filter_fn) {
        b = filter_fn(b);
      }
      auto res = detail::search(mapping, b, filter_fn);
      if (res.first) {
        input = detail::value_string(detail::pair_adaptor<element_t>::second(*res.second));
      }
      return std::string{};
    };
  }

  /// You can pass in as many filter functions as you like, they nest
  template <typename T, typename... Args>
  Transformer(T &&mapping, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&...other)
      : Transformer(
            std::forward<T>(mapping),
            [filter_fn_1, filter_fn_2](std::string a) { return filter_fn_2(filter_fn_1(a)); },
            other...) {}
};

/// translate named items to other or a value set
class CheckedTransformer : public Validator {
public:
  using filter_fn_t = std::function<std::string(std::string)>;

  /// This allows in-place construction
  template <typename... Args>
  CheckedTransformer(std::initializer_list<std::pair<std::string, std::string>> values, Args &&...args)
      : CheckedTransformer(TransformPairs<std::string>(values), std::forward<Args>(args)...) {}

  /// direct map of std::string to std::string
  template <typename T>
  explicit CheckedTransformer(T mapping) : CheckedTransformer(std::move(mapping), nullptr) {}

  /// This checks to see if an item is in a set: pointer or copy version. You can pass in a function that will filter
  /// both sides of the comparison before computing the comparison.
  template <typename T, typename F>
  explicit CheckedTransformer(T mapping, F filter_function) {

    static_assert(detail::pair_adaptor<typename detail::element_type<T>::type>::value,
                  "mapping must produce value pairs");
    // Get the type of the contained item - requires a container have ::value_type
    // if the type does not have first_type and second_type, these are both value_type
    using element_t = typename detail::element_type<T>::type;                      // Removes (smart) pointers if needed
    using item_t = typename detail::pair_adaptor<element_t>::first_type;           // Is value_type if not a map
    using local_item_t = typename IsMemberType<item_t>::type;                      // Will convert bad types to good ones
                                                                                   // (const char * to std::string)
    using iteration_type_t = typename detail::pair_adaptor<element_t>::value_type; // the type of the object pair

    // Make a local copy of the filter function, using a std::function if not one already
    std::function<local_item_t(local_item_t)> filter_fn = filter_function;

    auto tfunc = [mapping]() {
      std::string out("value in ");
      out += detail::generate_map(detail::smart_deref(mapping)) + " OR {";
      out += detail::join(
          detail::smart_deref(mapping),
          [](const iteration_type_t &v) { return detail::to_string(detail::pair_adaptor<element_t>::second(v)); },
          ",");
      out.push_back('}');
      return out;
    };

    desc_function_ = tfunc;

    func_ = [mapping, tfunc, filter_fn](std::string &input) {
      using CLI::detail::lexical_cast;
      local_item_t b;
      bool converted = lexical_cast(input, b);
      if (converted) {
        if (filter_fn) {
          b = filter_fn(b);
        }
        auto res = detail::search(mapping, b, filter_fn);
        if (res.first) {
          input = detail::value_string(detail::pair_adaptor<element_t>::second(*res.second));
          return std::string{};
        }
      }
      for (const auto &v : detail::smart_deref(mapping)) {
        auto output_string = detail::value_string(detail::pair_adaptor<element_t>::second(v));
        if (output_string == input) {
          return std::string();
        }
      }

      return "Check " + input + " " + tfunc() + " FAILED";
    };
  }

  /// You can pass in as many filter functions as you like, they nest
  template <typename T, typename... Args>
  CheckedTransformer(T &&mapping, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&...other)
      : CheckedTransformer(
            std::forward<T>(mapping),
            [filter_fn_1, filter_fn_2](std::string a) { return filter_fn_2(filter_fn_1(a)); },
            other...) {}
};

/// Helper function to allow ignore_case to be passed to IsMember or Transform
inline std::string ignore_case(std::string item) { return detail::to_lower(item); }

/// Helper function to allow ignore_underscore to be passed to IsMember or Transform
inline std::string ignore_underscore(std::string item) { return detail::remove_underscore(item); }

/// Helper function to allow checks to ignore spaces to be passed to IsMember or Transform
inline std::string ignore_space(std::string item) {
  item.erase(std::remove(std::begin(item), std::end(item), ' '), std::end(item));
  item.erase(std::remove(std::begin(item), std::end(item), '\t'), std::end(item));
  return item;
}

/// Multiply a number by a factor using given mapping.
/// Can be used to write transforms for SIZE or DURATION inputs.
///
/// Example:
///   With mapping = `{"b"->1, "kb"->1024, "mb"->1024*1024}`
///   one can recognize inputs like "100", "12kb", "100 MB",
///   that will be automatically transformed to 100, 14448, 104857600.
///
/// Output number type matches the type in the provided mapping.
/// Therefore, if it is required to interpret real inputs like "0.42 s",
/// the mapping should be of a type <string, float> or <string, double>.
class AsNumberWithUnit : public Validator {
public:
  /// Adjust AsNumberWithUnit behavior.
  /// CASE_SENSITIVE/CASE_INSENSITIVE controls how units are matched.
  /// UNIT_OPTIONAL/UNIT_REQUIRED throws ValidationError
  ///   if UNIT_REQUIRED is set and unit literal is not found.
  enum Options {
    CASE_SENSITIVE = 0,
    CASE_INSENSITIVE = 1,
    UNIT_OPTIONAL = 0,
    UNIT_REQUIRED = 2,
    DEFAULT = CASE_INSENSITIVE | UNIT_OPTIONAL
  };

  template <typename Number>
  explicit AsNumberWithUnit(std::map<std::string, Number> mapping,
                            Options opts = DEFAULT,
                            const std::string &unit_name = "UNIT") {
    description(generate_description<Number>(unit_name, opts));
    validate_mapping(mapping, opts);

    // transform function
    func_ = [mapping, opts](std::string &input) -> std::string {
      Number num{};

      detail::rtrim(input);
      if (input.empty()) {
        throw ValidationError("Input is empty");
      }

      // Find split position between number and prefix
      auto unit_begin = input.end();
      while (unit_begin > input.begin() && std::isalpha(*(unit_begin - 1), std::locale())) {
        --unit_begin;
      }

      std::string unit{unit_begin, input.end()};
      input.resize(static_cast<std::size_t>(std::distance(input.begin(), unit_begin)));
      detail::trim(input);

      if (opts & UNIT_REQUIRED && unit.empty()) {
        throw ValidationError("Missing mandatory unit");
      }
      if (opts & CASE_INSENSITIVE) {
        unit = detail::to_lower(unit);
      }
      if (unit.empty()) {
        using CLI::detail::lexical_cast;
        if (!lexical_cast(input, num)) {
          throw ValidationError(std::string("Value ") + input + " could not be converted to " +
                                detail::type_name<Number>());
        }
        // No need to modify input if no unit passed
        return {};
      }

      // find corresponding factor
      auto it = mapping.find(unit);
      if (it == mapping.end()) {
        throw ValidationError(unit +
                              " unit not recognized. "
                              "Allowed values: " +
                              detail::generate_map(mapping, true));
      }

      if (!input.empty()) {
        using CLI::detail::lexical_cast;
        bool converted = lexical_cast(input, num);
        if (!converted) {
          throw ValidationError(std::string("Value ") + input + " could not be converted to " +
                                detail::type_name<Number>());
        }
        // perform safe multiplication
        bool ok = detail::checked_multiply(num, it->second);
        if (!ok) {
          throw ValidationError(detail::to_string(num) + " multiplied by " + unit +
                                " factor would cause number overflow. Use smaller value.");
        }
      } else {
        num = static_cast<Number>(it->second);
      }

      input = detail::to_string(num);

      return {};
    };
  }

private:
  /// Check that mapping contains valid units.
  /// Update mapping for CASE_INSENSITIVE mode.
  template <typename Number>
  static void validate_mapping(std::map<std::string, Number> &mapping, Options opts) {
    for (auto &kv : mapping) {
      if (kv.first.empty()) {
        throw ValidationError("Unit must not be empty.");
      }
      if (!detail::isalpha(kv.first)) {
        throw ValidationError("Unit must contain only letters.");
      }
    }

    // make all units lowercase if CASE_INSENSITIVE
    if (opts & CASE_INSENSITIVE) {
      std::map<std::string, Number> lower_mapping;
      for (auto &kv : mapping) {
        auto s = detail::to_lower(kv.first);
        if (lower_mapping.count(s)) {
          throw ValidationError(std::string("Several matching lowercase unit representations are found: ") +
                                s);
        }
        lower_mapping[detail::to_lower(kv.first)] = kv.second;
      }
      mapping = std::move(lower_mapping);
    }
  }

  /// Generate description like this: NUMBER [UNIT]
  template <typename Number>
  static std::string generate_description(const std::string &name, Options opts) {
    std::stringstream out;
    out << detail::type_name<Number>() << ' ';
    if (opts & UNIT_REQUIRED) {
      out << name;
    } else {
      out << '[' << name << ']';
    }
    return out.str();
  }
};

inline AsNumberWithUnit::Options operator|(const AsNumberWithUnit::Options &a, const AsNumberWithUnit::Options &b) {
  return static_cast<AsNumberWithUnit::Options>(static_cast<int>(a) | static_cast<int>(b));
}

/// Converts a human-readable size string (with unit literal) to uin64_t size.
/// Example:
///   "100" => 100
///   "1 b" => 100
///   "10Kb" => 10240 // you can configure this to be interpreted as kilobyte (*1000) or kibibyte (*1024)
///   "10 KB" => 10240
///   "10 kb" => 10240
///   "10 kib" => 10240 // *i, *ib are always interpreted as *bibyte (*1024)
///   "10kb" => 10240
///   "2 MB" => 2097152
///   "2 EiB" => 2^61 // Units up to exibyte are supported
class AsSizeValue : public AsNumberWithUnit {
public:
  using result_t = std::uint64_t;

  /// If kb_is_1000 is true,
  /// interpret 'kb', 'k' as 1000 and 'kib', 'ki' as 1024
  /// (same applies to higher order units as well).
  /// Otherwise, interpret all literals as factors of 1024.
  /// The first option is formally correct, but
  /// the second interpretation is more wide-spread
  /// (see https://en.wikipedia.org/wiki/Binary_prefix).
  explicit AsSizeValue(bool kb_is_1000);

private:
  /// Get <size unit, factor> mapping
  static std::map<std::string, result_t> init_mapping(bool kb_is_1000);

  /// Cache calculated mapping
  static std::map<std::string, result_t> get_mapping(bool kb_is_1000);
};

namespace detail {
/// Split a string into a program name and command line arguments
/// the string is assumed to contain a file name followed by other arguments
/// the return value contains is a pair with the first argument containing the program name and the second
/// everything else.
CLI11_INLINE std::pair<std::string, std::string> split_program_name(std::string commandline);

} // namespace detail
/// @}

CLI11_INLINE std::string Validator::operator()(std::string &str) const {
  std::string retstring;
  if (active_) {
    if (non_modifying_) {
      std::string value = str;
      retstring = func_(value);
    } else {
      retstring = func_(str);
    }
  }
  return retstring;
}

CLI11_NODISCARD CLI11_INLINE Validator Validator::description(std::string validator_desc) const {
  Validator newval(*this);
  newval.desc_function_ = [validator_desc]() { return validator_desc; };
  return newval;
}

CLI11_INLINE Validator Validator::operator&(const Validator &other) const {
  Validator newval;

  newval._merge_description(*this, other, " AND ");

  // Give references (will make a copy in lambda function)
  const std::function<std::string(std::string & filename)> &f1 = func_;
  const std::function<std::string(std::string & filename)> &f2 = other.func_;

  newval.func_ = [f1, f2](std::string &input) {
    std::string s1 = f1(input);
    std::string s2 = f2(input);
    if (!s1.empty() && !s2.empty())
      return std::string("(") + s1 + ") AND (" + s2 + ")";
    return s1 + s2;
  };

  newval.active_ = active_ && other.active_;
  newval.application_index_ = application_index_;
  return newval;
}

CLI11_INLINE Validator Validator::operator|(const Validator &other) const {
  Validator newval;

  newval._merge_description(*this, other, " OR ");

  // Give references (will make a copy in lambda function)
  const std::function<std::string(std::string &)> &f1 = func_;
  const std::function<std::string(std::string &)> &f2 = other.func_;

  newval.func_ = [f1, f2](std::string &input) {
    std::string s1 = f1(input);
    std::string s2 = f2(input);
    if (s1.empty() || s2.empty())
      return std::string();

    return std::string("(") + s1 + ") OR (" + s2 + ")";
  };
  newval.active_ = active_ && other.active_;
  newval.application_index_ = application_index_;
  return newval;
}

CLI11_INLINE Validator Validator::operator!() const {
  Validator newval;
  const std::function<std::string()> &dfunc1 = desc_function_;
  newval.desc_function_ = [dfunc1]() {
    auto str = dfunc1();
    return (!str.empty()) ? std::string("NOT ") + str : std::string{};
  };
  // Give references (will make a copy in lambda function)
  const std::function<std::string(std::string & res)> &f1 = func_;

  newval.func_ = [f1, dfunc1](std::string &test) -> std::string {
    std::string s1 = f1(test);
    if (s1.empty()) {
      return std::string("check ") + dfunc1() + " succeeded improperly";
    }
    return std::string{};
  };
  newval.active_ = active_;
  newval.application_index_ = application_index_;
  return newval;
}

CLI11_INLINE void
Validator::_merge_description(const Validator &val1, const Validator &val2, const std::string &merger) {

  const std::function<std::string()> &dfunc1 = val1.desc_function_;
  const std::function<std::string()> &dfunc2 = val2.desc_function_;

  desc_function_ = [=]() {
    std::string f1 = dfunc1();
    std::string f2 = dfunc2();
    if ((f1.empty()) || (f2.empty())) {
      return f1 + f2;
    }
    return std::string(1, '(') + f1 + ')' + merger + '(' + f2 + ')';
  };
}

namespace detail {

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
CLI11_INLINE path_type check_path(const char *file) noexcept {
  std::error_code ec;
  auto stat = std::filesystem::status(to_path(file), ec);
  if (ec) {
    return path_type::nonexistent;
  }
  switch (stat.type()) {
  case std::filesystem::file_type::none: // LCOV_EXCL_LINE
  case std::filesystem::file_type::not_found:
    return path_type::nonexistent; // LCOV_EXCL_LINE
  case std::filesystem::file_type::directory:
    return path_type::directory;
  case std::filesystem::file_type::symlink:
  case std::filesystem::file_type::block:
  case std::filesystem::file_type::character:
  case std::filesystem::file_type::fifo:
  case std::filesystem::file_type::socket:
  case std::filesystem::file_type::regular:
  case std::filesystem::file_type::unknown:
  default:
    return path_type::file;
  }
}
#else
CLI11_INLINE path_type check_path(const char *file) noexcept {
#if defined(_MSC_VER)
  struct __stat64 buffer;
  if (_stat64(file, &buffer) == 0) {
    return ((buffer.st_mode & S_IFDIR) != 0) ? path_type::directory : path_type::file;
  }
#else
  struct stat buffer;
  if (stat(file, &buffer) == 0) {
    return ((buffer.st_mode & S_IFDIR) != 0) ? path_type::directory : path_type::file;
  }
#endif
  return path_type::nonexistent;
}
#endif

CLI11_INLINE ExistingFileValidator::ExistingFileValidator() : Validator("FILE") {
  func_ = [](std::string &filename) {
    auto path_result = check_path(filename.c_str());
    if (path_result == path_type::nonexistent) {
      return "File does not exist: " + filename;
    }
    if (path_result == path_type::directory) {
      return "File is actually a directory: " + filename;
    }
    return std::string();
  };
}

CLI11_INLINE ExistingDirectoryValidator::ExistingDirectoryValidator() : Validator("DIR") {
  func_ = [](std::string &filename) {
    auto path_result = check_path(filename.c_str());
    if (path_result == path_type::nonexistent) {
      return "Directory does not exist: " + filename;
    }
    if (path_result == path_type::file) {
      return "Directory is actually a file: " + filename;
    }
    return std::string();
  };
}

CLI11_INLINE ExistingPathValidator::ExistingPathValidator() : Validator("PATH(existing)") {
  func_ = [](std::string &filename) {
    auto path_result = check_path(filename.c_str());
    if (path_result == path_type::nonexistent) {
      return "Path does not exist: " + filename;
    }
    return std::string();
  };
}

CLI11_INLINE NonexistentPathValidator::NonexistentPathValidator() : Validator("PATH(non-existing)") {
  func_ = [](std::string &filename) {
    auto path_result = check_path(filename.c_str());
    if (path_result != path_type::nonexistent) {
      return "Path already exists: " + filename;
    }
    return std::string();
  };
}

CLI11_INLINE IPV4Validator::IPV4Validator() : Validator("IPV4") {
  func_ = [](std::string &ip_addr) {
    auto result = CLI::detail::split(ip_addr, '.');
    if (result.size() != 4) {
      return std::string("Invalid IPV4 address must have four parts (") + ip_addr + ')';
    }
    int num = 0;
    for (const auto &var : result) {
      using CLI::detail::lexical_cast;
      bool retval = lexical_cast(var, num);
      if (!retval) {
        return std::string("Failed parsing number (") + var + ')';
      }
      if (num < 0 || num > 255) {
        return std::string("Each IP number must be between 0 and 255 ") + var;
      }
    }
    return std::string{};
  };
}

CLI11_INLINE EscapedStringTransformer::EscapedStringTransformer() {
  func_ = [](std::string &str) {
    try {
      if (str.size() > 1 && (str.front() == '\"' || str.front() == '\'' || str.front() == '`') &&
          str.front() == str.back()) {
        process_quoted_string(str);
      } else if (str.find_first_of('\\') != std::string::npos) {
        if (detail::is_binary_escaped_string(str)) {
          str = detail::extract_binary_string(str);
        } else {
          str = remove_escaped_characters(str);
        }
      }
      return std::string{};
    } catch (const std::invalid_argument &ia) {
      return std::string(ia.what());
    }
  };
}
} // namespace detail

CLI11_INLINE FileOnDefaultPath::FileOnDefaultPath(std::string default_path, bool enableErrorReturn)
    : Validator("FILE") {
  func_ = [default_path, enableErrorReturn](std::string &filename) {
    auto path_result = detail::check_path(filename.c_str());
    if (path_result == detail::path_type::nonexistent) {
      std::string test_file_path = default_path;
      if (default_path.back() != '/' && default_path.back() != '\\') {
        // Add folder separator
        test_file_path += '/';
      }
      test_file_path.append(filename);
      path_result = detail::check_path(test_file_path.c_str());
      if (path_result == detail::path_type::file) {
        filename = test_file_path;
      } else {
        if (enableErrorReturn) {
          return "File does not exist: " + filename;
        }
      }
    }
    return std::string{};
  };
}

CLI11_INLINE AsSizeValue::AsSizeValue(bool kb_is_1000) : AsNumberWithUnit(get_mapping(kb_is_1000)) {
  if (kb_is_1000) {
    description("SIZE [b, kb(=1000b), kib(=1024b), ...]");
  } else {
    description("SIZE [b, kb(=1024b), ...]");
  }
}

CLI11_INLINE std::map<std::string, AsSizeValue::result_t> AsSizeValue::init_mapping(bool kb_is_1000) {
  std::map<std::string, result_t> m;
  result_t k_factor = kb_is_1000 ? 1000 : 1024;
  result_t ki_factor = 1024;
  result_t k = 1;
  result_t ki = 1;
  m["b"] = 1;
  for (std::string p : {"k", "m", "g", "t", "p", "e"}) {
    k *= k_factor;
    ki *= ki_factor;
    m[p] = k;
    m[p + "b"] = k;
    m[p + "i"] = ki;
    m[p + "ib"] = ki;
  }
  return m;
}

CLI11_INLINE std::map<std::string, AsSizeValue::result_t> AsSizeValue::get_mapping(bool kb_is_1000) {
  if (kb_is_1000) {
    static auto m = init_mapping(true);
    return m;
  }
  static auto m = init_mapping(false);
  return m;
}

namespace detail {

CLI11_INLINE std::pair<std::string, std::string> split_program_name(std::string commandline) {
  // try to determine the programName
  std::pair<std::string, std::string> vals;
  trim(commandline);
  auto esp = commandline.find_first_of(' ', 1);
  while (detail::check_path(commandline.substr(0, esp).c_str()) != path_type::file) {
    esp = commandline.find_first_of(' ', esp + 1);
    if (esp == std::string::npos) {
      // if we have reached the end and haven't found a valid file just assume the first argument is the
      // program name
      if (commandline[0] == '"' || commandline[0] == '\'' || commandline[0] == '`') {
        bool embeddedQuote = false;
        auto keyChar = commandline[0];
        auto end = commandline.find_first_of(keyChar, 1);
        while ((end != std::string::npos) && (commandline[end - 1] == '\\')) { // deal with escaped quotes
          end = commandline.find_first_of(keyChar, end + 1);
          embeddedQuote = true;
        }
        if (end != std::string::npos) {
          vals.first = commandline.substr(1, end - 1);
          esp = end + 1;
          if (embeddedQuote) {
            vals.first = find_and_replace(vals.first, std::string("\\") + keyChar, std::string(1, keyChar));
          }
        } else {
          esp = commandline.find_first_of(' ', 1);
        }
      } else {
        esp = commandline.find_first_of(' ', 1);
      }

      break;
    }
  }
  if (vals.first.empty()) {
    vals.first = commandline.substr(0, esp);
    rtrim(vals.first);
  }

  // strip the program name
  vals.second = (esp < commandline.length() - 1) ? commandline.substr(esp + 1) : std::string{};
  ltrim(vals.second);
  return vals;
}

} // namespace detail
/// @}

class Option;
class App;

/// This enum signifies the type of help requested
///
/// This is passed in by App; all user classes must accept this as
/// the second argument.

enum class AppFormatMode {
  Normal, ///< The normal, detailed help
  All,    ///< A fully expanded help
  Sub,    ///< Used when printed as part of expanded subcommand
};

/// This is the minimum requirements to run a formatter.
///
/// A user can subclass this is if they do not care at all
/// about the structure in CLI::Formatter.
class FormatterBase {
protected:
  /// @name Options
  ///@{

  /// The width of the first column
  std::size_t column_width_{30};

  /// @brief The required help printout labels (user changeable)
  /// Values are Needs, Excludes, etc.
  std::map<std::string, std::string> labels_{};

  ///@}
  /// @name Basic
  ///@{

public:
  FormatterBase() = default;
  FormatterBase(const FormatterBase &) = default;
  FormatterBase(FormatterBase &&) = default;
  FormatterBase &operator=(const FormatterBase &) = default;
  FormatterBase &operator=(FormatterBase &&) = default;

  /// Adding a destructor in this form to work around bug in GCC 4.7
  virtual ~FormatterBase() noexcept {} // NOLINT(modernize-use-equals-default)

  /// This is the key method that puts together help
  virtual std::string make_help(const App *, std::string, AppFormatMode) const = 0;

  ///@}
  /// @name Setters
  ///@{

  /// Set the "REQUIRED" label
  void label(std::string key, std::string val) { labels_[key] = val; }

  /// Set the column width
  void column_width(std::size_t val) { column_width_ = val; }

  ///@}
  /// @name Getters
  ///@{

  /// Get the current value of a name (REQUIRED, etc.)
  CLI11_NODISCARD std::string get_label(std::string key) const {
    if (labels_.find(key) == labels_.end())
      return key;
    return labels_.at(key);
  }

  /// Get the current column width
  CLI11_NODISCARD std::size_t get_column_width() const { return column_width_; }

  ///@}
};

/// This is a specialty override for lambda functions
class FormatterLambda final : public FormatterBase {
  using funct_t = std::function<std::string(const App *, std::string, AppFormatMode)>;

  /// The lambda to hold and run
  funct_t lambda_;

public:
  /// Create a FormatterLambda with a lambda function
  explicit FormatterLambda(funct_t funct) : lambda_(std::move(funct)) {}

  /// Adding a destructor (mostly to make GCC 4.7 happy)
  ~FormatterLambda() noexcept override {} // NOLINT(modernize-use-equals-default)

  /// This will simply call the lambda function
  std::string make_help(const App *app, std::string name, AppFormatMode mode) const override {
    return lambda_(app, name, mode);
  }
};

/// This is the default Formatter for CLI11. It pretty prints help output, and is broken into quite a few
/// overridable methods, to be highly customizable with minimal effort.
class Formatter : public FormatterBase {
public:
  Formatter() = default;
  Formatter(const Formatter &) = default;
  Formatter(Formatter &&) = default;
  Formatter &operator=(const Formatter &) = default;
  Formatter &operator=(Formatter &&) = default;

  /// @name Overridables
  ///@{

  /// This prints out a group of options with title
  ///
  CLI11_NODISCARD virtual std::string
  make_group(std::string group, bool is_positional, std::vector<const Option *> opts) const;

  /// This prints out just the positionals "group"
  virtual std::string make_positionals(const App *app) const;

  /// This prints out all the groups of options
  std::string make_groups(const App *app, AppFormatMode mode) const;

  /// This prints out all the subcommands
  virtual std::string make_subcommands(const App *app, AppFormatMode mode) const;

  /// This prints out a subcommand
  virtual std::string make_subcommand(const App *sub) const;

  /// This prints out a subcommand in help-all
  virtual std::string make_expanded(const App *sub) const;

  /// This prints out all the groups of options
  virtual std::string make_footer(const App *app) const;

  /// This displays the description line
  virtual std::string make_description(const App *app) const;

  /// This displays the usage line
  virtual std::string make_usage(const App *app, std::string name) const;

  /// This puts everything together
  std::string make_help(const App * /*app*/, std::string, AppFormatMode) const override;

  ///@}
  /// @name Options
  ///@{

  /// This prints out an option help line, either positional or optional form
  virtual std::string make_option(const Option *opt, bool is_positional) const {
    std::stringstream out;
    detail::format_help(
        out, make_option_name(opt, is_positional) + make_option_opts(opt), make_option_desc(opt), column_width_);
    return out.str();
  }

  /// @brief This is the name part of an option, Default: left column
  virtual std::string make_option_name(const Option *, bool) const;

  /// @brief This is the options part of the name, Default: combined into left column
  virtual std::string make_option_opts(const Option *) const;

  /// @brief This is the description. Default: Right column, on new line if left column too large
  virtual std::string make_option_desc(const Option *) const;

  /// @brief This is used to print the name on the USAGE line
  virtual std::string make_option_usage(const Option *opt) const;

  ///@}
};

using results_t = std::vector<std::string>;
/// callback function definition
using callback_t = std::function<bool(const results_t &)>;

class Option;
class App;

using Option_p = std::unique_ptr<Option>;
/// Enumeration of the multiOption Policy selection
enum class MultiOptionPolicy : char {
  Throw,     //!< Throw an error if any extra arguments were given
  TakeLast,  //!< take only the last Expected number of arguments
  TakeFirst, //!< take only the first Expected number of arguments
  Join,      //!< merge all the arguments together into a single string via the delimiter character default('\n')
  TakeAll,   //!< just get all the passed argument regardless
  Sum,       //!< sum all the arguments together if numerical or concatenate directly without delimiter
  Reverse,   //!< take only the last Expected number of arguments in reverse order
};

/// This is the CRTP base class for Option and OptionDefaults. It was designed this way
/// to share parts of the class; an OptionDefaults can copy to an Option.
template <typename CRTP>
class OptionBase {
  friend App;

protected:
  /// The group membership
  std::string group_ = std::string("Options");

  /// True if this is a required option
  bool required_{false};

  /// Ignore the case when matching (option, not value)
  bool ignore_case_{false};

  /// Ignore underscores when matching (option, not value)
  bool ignore_underscore_{false};

  /// Allow this option to be given in a configuration file
  bool configurable_{true};

  /// Disable overriding flag values with '=value'
  bool disable_flag_override_{false};

  /// Specify a delimiter character for vector arguments
  char delimiter_{'\0'};

  /// Automatically capture default value
  bool always_capture_default_{false};

  /// Policy for handling multiple arguments beyond the expected Max
  MultiOptionPolicy multi_option_policy_{MultiOptionPolicy::Throw};

  /// Copy the contents to another similar class (one based on OptionBase)
  template <typename T>
  void copy_to(T *other) const;

public:
  // setters

  /// Changes the group membership
  CRTP *group(const std::string &name) {
    if (!detail::valid_alias_name_string(name)) {
      throw IncorrectConstruction("Group names may not contain newlines or null characters");
    }
    group_ = name;
    return static_cast<CRTP *>(this);
  }

  /// Set the option as required
  CRTP *required(bool value = true) {
    required_ = value;
    return static_cast<CRTP *>(this);
  }

  /// Support Plumbum term
  CRTP *mandatory(bool value = true) { return required(value); }

  CRTP *always_capture_default(bool value = true) {
    always_capture_default_ = value;
    return static_cast<CRTP *>(this);
  }

  // Getters

  /// Get the group of this option
  CLI11_NODISCARD const std::string &get_group() const { return group_; }

  /// True if this is a required option
  CLI11_NODISCARD bool get_required() const { return required_; }

  /// The status of ignore case
  CLI11_NODISCARD bool get_ignore_case() const { return ignore_case_; }

  /// The status of ignore_underscore
  CLI11_NODISCARD bool get_ignore_underscore() const { return ignore_underscore_; }

  /// The status of configurable
  CLI11_NODISCARD bool get_configurable() const { return configurable_; }

  /// The status of configurable
  CLI11_NODISCARD bool get_disable_flag_override() const { return disable_flag_override_; }

  /// Get the current delimiter char
  CLI11_NODISCARD char get_delimiter() const { return delimiter_; }

  /// Return true if this will automatically capture the default value for help printing
  CLI11_NODISCARD bool get_always_capture_default() const { return always_capture_default_; }

  /// The status of the multi option policy
  CLI11_NODISCARD MultiOptionPolicy get_multi_option_policy() const { return multi_option_policy_; }

  // Shortcuts for multi option policy

  /// Set the multi option policy to take last
  CRTP *take_last() {
    auto *self = static_cast<CRTP *>(this);
    self->multi_option_policy(MultiOptionPolicy::TakeLast);
    return self;
  }

  /// Set the multi option policy to take last
  CRTP *take_first() {
    auto *self = static_cast<CRTP *>(this);
    self->multi_option_policy(MultiOptionPolicy::TakeFirst);
    return self;
  }

  /// Set the multi option policy to take all arguments
  CRTP *take_all() {
    auto self = static_cast<CRTP *>(this);
    self->multi_option_policy(MultiOptionPolicy::TakeAll);
    return self;
  }

  /// Set the multi option policy to join
  CRTP *join() {
    auto *self = static_cast<CRTP *>(this);
    self->multi_option_policy(MultiOptionPolicy::Join);
    return self;
  }

  /// Set the multi option policy to join with a specific delimiter
  CRTP *join(char delim) {
    auto self = static_cast<CRTP *>(this);
    self->delimiter_ = delim;
    self->multi_option_policy(MultiOptionPolicy::Join);
    return self;
  }

  /// Allow in a configuration file
  CRTP *configurable(bool value = true) {
    configurable_ = value;
    return static_cast<CRTP *>(this);
  }

  /// Allow in a configuration file
  CRTP *delimiter(char value = '\0') {
    delimiter_ = value;
    return static_cast<CRTP *>(this);
  }
};

/// This is a version of OptionBase that only supports setting values,
/// for defaults. It is stored as the default option in an App.
class OptionDefaults : public OptionBase<OptionDefaults> {
public:
  OptionDefaults() = default;

  // Methods here need a different implementation if they are Option vs. OptionDefault

  /// Take the last argument if given multiple times
  OptionDefaults *multi_option_policy(MultiOptionPolicy value = MultiOptionPolicy::Throw) {
    multi_option_policy_ = value;
    return this;
  }

  /// Ignore the case of the option name
  OptionDefaults *ignore_case(bool value = true) {
    ignore_case_ = value;
    return this;
  }

  /// Ignore underscores in the option name
  OptionDefaults *ignore_underscore(bool value = true) {
    ignore_underscore_ = value;
    return this;
  }

  /// Disable overriding flag values with an '=<value>' segment
  OptionDefaults *disable_flag_override(bool value = true) {
    disable_flag_override_ = value;
    return this;
  }

  /// set a delimiter character to split up single arguments to treat as multiple inputs
  OptionDefaults *delimiter(char value = '\0') {
    delimiter_ = value;
    return this;
  }
};

class Option : public OptionBase<Option> {
  friend App;

protected:
  /// @name Names
  ///@{

  /// A list of the short names (`-a`) without the leading dashes
  std::vector<std::string> snames_{};

  /// A list of the long names (`--long`) without the leading dashes
  std::vector<std::string> lnames_{};

  /// A list of the flag names with the appropriate default value, the first part of the pair should be duplicates of
  /// what is in snames or lnames but will trigger a particular response on a flag
  std::vector<std::pair<std::string, std::string>> default_flag_values_{};

  /// a list of flag names with specified default values;
  std::vector<std::string> fnames_{};

  /// A positional name
  std::string pname_{};

  /// If given, check the environment for this option
  std::string envname_{};

  ///@}
  /// @name Help
  ///@{

  /// The description for help strings
  std::string description_{};

  /// A human readable default value, either manually set, captured, or captured by default
  std::string default_str_{};

  /// If given, replace the text that describes the option type and usage in the help text
  std::string option_text_{};

  /// A human readable type value, set when App creates this
  ///
  /// This is a lambda function so "types" can be dynamic, such as when a set prints its contents.
  std::function<std::string()> type_name_{[]() { return std::string(); }};

  /// Run this function to capture a default (ignore if empty)
  std::function<std::string()> default_function_{};

  ///@}
  /// @name Configuration
  ///@{

  /// The number of arguments that make up one option. max is the nominal type size, min is the minimum number of
  /// strings
  int type_size_max_{1};
  /// The minimum number of arguments an option should be expecting
  int type_size_min_{1};

  /// The minimum number of expected values
  int expected_min_{1};
  /// The maximum number of expected values
  int expected_max_{1};

  /// A list of Validators to run on each value parsed
  std::vector<Validator> validators_{};

  /// A list of options that are required with this option
  std::set<Option *> needs_{};

  /// A list of options that are excluded with this option
  std::set<Option *> excludes_{};

  ///@}
  /// @name Other
  ///@{

  /// link back up to the parent App for fallthrough
  App *parent_{nullptr};

  /// Options store a callback to do all the work
  callback_t callback_{};

  ///@}
  /// @name Parsing results
  ///@{

  /// complete Results of parsing
  results_t results_{};
  /// results after reduction
  results_t proc_results_{};
  /// enumeration for the option state machine
  enum class option_state : char {
    parsing = 0,      //!< The option is currently collecting parsed results
    validated = 2,    //!< the results have been validated
    reduced = 4,      //!< a subset of results has been generated
    callback_run = 6, //!< the callback has been executed
  };
  /// Whether the callback has run (needed for INI parsing)
  option_state current_option_state_{option_state::parsing};
  /// Specify that extra args beyond type_size_max should be allowed
  bool allow_extra_args_{false};
  /// Specify that the option should act like a flag vs regular option
  bool flag_like_{false};
  /// Control option to run the callback to set the default
  bool run_callback_for_default_{false};
  /// flag indicating a separator needs to be injected after each argument call
  bool inject_separator_{false};
  /// flag indicating that the option should trigger the validation and callback chain on each result when loaded
  bool trigger_on_result_{false};
  /// flag indicating that the option should force the callback regardless if any results present
  bool force_callback_{false};
  ///@}

  /// Making an option by hand is not defined, it must be made by the App class
  Option(std::string option_name, std::string option_description, callback_t callback, App *parent)
      : description_(std::move(option_description)), parent_(parent), callback_(std::move(callback)) {
    std::tie(snames_, lnames_, pname_) = detail::get_names(detail::split_names(option_name));
  }

public:
  /// @name Basic
  ///@{

  Option(const Option &) = delete;
  Option &operator=(const Option &) = delete;

  /// Count the total number of times an option was passed
  CLI11_NODISCARD std::size_t count() const { return results_.size(); }

  /// True if the option was not passed
  CLI11_NODISCARD bool empty() const { return results_.empty(); }

  /// This bool operator returns true if any arguments were passed or the option callback is forced
  explicit operator bool() const { return !empty() || force_callback_; }

  /// Clear the parsed results (mostly for testing)
  void clear() {
    results_.clear();
    current_option_state_ = option_state::parsing;
  }

  ///@}
  /// @name Setting options
  ///@{

  /// Set the number of expected arguments
  Option *expected(int value);

  /// Set the range of expected arguments
  Option *expected(int value_min, int value_max);

  /// Set the value of allow_extra_args which allows extra value arguments on the flag or option to be included
  /// with each instance
  Option *allow_extra_args(bool value = true) {
    allow_extra_args_ = value;
    return this;
  }
  /// Get the current value of allow extra args
  CLI11_NODISCARD bool get_allow_extra_args() const { return allow_extra_args_; }
  /// Set the value of trigger_on_parse which specifies that the option callback should be triggered on every parse
  Option *trigger_on_parse(bool value = true) {
    trigger_on_result_ = value;
    return this;
  }
  /// The status of trigger on parse
  CLI11_NODISCARD bool get_trigger_on_parse() const { return trigger_on_result_; }

  /// Set the value of force_callback
  Option *force_callback(bool value = true) {
    force_callback_ = value;
    return this;
  }
  /// The status of force_callback
  CLI11_NODISCARD bool get_force_callback() const { return force_callback_; }

  /// Set the value of run_callback_for_default which controls whether the callback function should be called to set
  /// the default This is controlled automatically but could be manipulated by the user.
  Option *run_callback_for_default(bool value = true) {
    run_callback_for_default_ = value;
    return this;
  }
  /// Get the current value of run_callback_for_default
  CLI11_NODISCARD bool get_run_callback_for_default() const { return run_callback_for_default_; }

  /// Adds a Validator with a built in type name
  Option *check(Validator validator, const std::string &validator_name = "");

  /// Adds a Validator. Takes a const string& and returns an error message (empty if conversion/check is okay).
  Option *check(std::function<std::string(const std::string &)> Validator,
                std::string Validator_description = "",
                std::string Validator_name = "");

  /// Adds a transforming Validator with a built in type name
  Option *transform(Validator Validator, const std::string &Validator_name = "");

  /// Adds a Validator-like function that can change result
  Option *transform(const std::function<std::string(std::string)> &func,
                    std::string transform_description = "",
                    std::string transform_name = "");

  /// Adds a user supplied function to run on each item passed in (communicate though lambda capture)
  Option *each(const std::function<void(std::string)> &func);

  /// Get a named Validator
  Validator *get_validator(const std::string &Validator_name = "");

  /// Get a Validator by index NOTE: this may not be the order of definition
  Validator *get_validator(int index);

  /// Sets required options
  Option *needs(Option *opt) {
    if (opt != this) {
      needs_.insert(opt);
    }
    return this;
  }

  /// Can find a string if needed
  template <typename T = App>
  Option *needs(std::string opt_name) {
    auto opt = static_cast<T *>(parent_)->get_option_no_throw(opt_name);
    if (opt == nullptr) {
      throw IncorrectConstruction::MissingOption(opt_name);
    }
    return needs(opt);
  }

  /// Any number supported, any mix of string and Opt
  template <typename A, typename B, typename... ARG>
  Option *needs(A opt, B opt1, ARG... args) {
    needs(opt);
    return needs(opt1, args...); // NOLINT(readability-suspicious-call-argument)
  }

  /// Remove needs link from an option. Returns true if the option really was in the needs list.
  bool remove_needs(Option *opt);

  /// Sets excluded options
  Option *excludes(Option *opt);

  /// Can find a string if needed
  template <typename T = App>
  Option *excludes(std::string opt_name) {
    auto opt = static_cast<T *>(parent_)->get_option_no_throw(opt_name);
    if (opt == nullptr) {
      throw IncorrectConstruction::MissingOption(opt_name);
    }
    return excludes(opt);
  }

  /// Any number supported, any mix of string and Opt
  template <typename A, typename B, typename... ARG>
  Option *excludes(A opt, B opt1, ARG... args) {
    excludes(opt);
    return excludes(opt1, args...);
  }

  /// Remove needs link from an option. Returns true if the option really was in the needs list.
  bool remove_excludes(Option *opt);

  /// Sets environment variable to read if no option given
  Option *envname(std::string name) {
    envname_ = std::move(name);
    return this;
  }

  /// Ignore case
  ///
  /// The template hides the fact that we don't have the definition of App yet.
  /// You are never expected to add an argument to the template here.
  template <typename T = App>
  Option *ignore_case(bool value = true);

  /// Ignore underscores in the option names
  ///
  /// The template hides the fact that we don't have the definition of App yet.
  /// You are never expected to add an argument to the template here.
  template <typename T = App>
  Option *ignore_underscore(bool value = true);

  /// Take the last argument if given multiple times (or another policy)
  Option *multi_option_policy(MultiOptionPolicy value = MultiOptionPolicy::Throw);

  /// Disable flag overrides values, e.g. --flag=<value> is not allowed
  Option *disable_flag_override(bool value = true) {
    disable_flag_override_ = value;
    return this;
  }
  ///@}
  /// @name Accessors
  ///@{

  /// The number of arguments the option expects
  CLI11_NODISCARD int get_type_size() const { return type_size_min_; }

  /// The minimum number of arguments the option expects
  CLI11_NODISCARD int get_type_size_min() const { return type_size_min_; }
  /// The maximum number of arguments the option expects
  CLI11_NODISCARD int get_type_size_max() const { return type_size_max_; }

  /// Return the inject_separator flag
  CLI11_NODISCARD bool get_inject_separator() const { return inject_separator_; }

  /// The environment variable associated to this value
  CLI11_NODISCARD std::string get_envname() const { return envname_; }

  /// The set of options needed
  CLI11_NODISCARD std::set<Option *> get_needs() const { return needs_; }

  /// The set of options excluded
  CLI11_NODISCARD std::set<Option *> get_excludes() const { return excludes_; }

  /// The default value (for help printing)
  CLI11_NODISCARD std::string get_default_str() const { return default_str_; }

  /// Get the callback function
  CLI11_NODISCARD callback_t get_callback() const { return callback_; }

  /// Get the long names
  CLI11_NODISCARD const std::vector<std::string> &get_lnames() const { return lnames_; }

  /// Get the short names
  CLI11_NODISCARD const std::vector<std::string> &get_snames() const { return snames_; }

  /// Get the flag names with specified default values
  CLI11_NODISCARD const std::vector<std::string> &get_fnames() const { return fnames_; }
  /// Get a single name for the option, first of lname, pname, sname, envname
  CLI11_NODISCARD const std::string &get_single_name() const {
    if (!lnames_.empty()) {
      return lnames_[0];
    }
    if (!snames_.empty()) {
      return snames_[0];
    }
    if (!pname_.empty()) {
      return pname_;
    }
    return envname_;
  }
  /// The number of times the option expects to be included
  CLI11_NODISCARD int get_expected() const { return expected_min_; }

  /// The number of times the option expects to be included
  CLI11_NODISCARD int get_expected_min() const { return expected_min_; }
  /// The max number of times the option expects to be included
  CLI11_NODISCARD int get_expected_max() const { return expected_max_; }

  /// The total min number of expected  string values to be used
  CLI11_NODISCARD int get_items_expected_min() const { return type_size_min_ * expected_min_; }

  /// Get the maximum number of items expected to be returned and used for the callback
  CLI11_NODISCARD int get_items_expected_max() const {
    int t = type_size_max_;
    return detail::checked_multiply(t, expected_max_) ? t : detail::expected_max_vector_size;
  }
  /// The total min number of expected  string values to be used
  CLI11_NODISCARD int get_items_expected() const { return get_items_expected_min(); }

  /// True if the argument can be given directly
  CLI11_NODISCARD bool get_positional() const { return !pname_.empty(); }

  /// True if option has at least one non-positional name
  CLI11_NODISCARD bool nonpositional() const { return (!lnames_.empty() || !snames_.empty()); }

  /// True if option has description
  CLI11_NODISCARD bool has_description() const { return !description_.empty(); }

  /// Get the description
  CLI11_NODISCARD const std::string &get_description() const { return description_; }

  /// Set the description
  Option *description(std::string option_description) {
    description_ = std::move(option_description);
    return this;
  }

  Option *option_text(std::string text) {
    option_text_ = std::move(text);
    return this;
  }

  CLI11_NODISCARD const std::string &get_option_text() const { return option_text_; }

  ///@}
  /// @name Help tools
  ///@{

  /// \brief Gets a comma separated list of names.
  /// Will include / prefer the positional name if positional is true.
  /// If all_options is false, pick just the most descriptive name to show.
  /// Use `get_name(true)` to get the positional name (replaces `get_pname`)
  CLI11_NODISCARD std::string get_name(bool positional = false, ///< Show the positional name
                                       bool all_options = false ///< Show every option
  ) const;

  ///@}
  /// @name Parser tools
  ///@{

  /// Process the callback
  void run_callback();

  /// If options share any of the same names, find it
  CLI11_NODISCARD const std::string &matching_name(const Option &other) const;

  /// If options share any of the same names, they are equal (not counting positional)
  bool operator==(const Option &other) const { return !matching_name(other).empty(); }

  /// Check a name. Requires "-" or "--" for short / long, supports positional name
  CLI11_NODISCARD bool check_name(const std::string &name) const;

  /// Requires "-" to be removed from string
  CLI11_NODISCARD bool check_sname(std::string name) const {
    return (detail::find_member(std::move(name), snames_, ignore_case_) >= 0);
  }

  /// Requires "--" to be removed from string
  CLI11_NODISCARD bool check_lname(std::string name) const {
    return (detail::find_member(std::move(name), lnames_, ignore_case_, ignore_underscore_) >= 0);
  }

  /// Requires "--" to be removed from string
  CLI11_NODISCARD bool check_fname(std::string name) const {
    if (fnames_.empty()) {
      return false;
    }
    return (detail::find_member(std::move(name), fnames_, ignore_case_, ignore_underscore_) >= 0);
  }

  /// Get the value that goes for a flag, nominally gets the default value but allows for overrides if not
  /// disabled
  CLI11_NODISCARD std::string get_flag_value(const std::string &name, std::string input_value) const;

  /// Puts a result at the end
  Option *add_result(std::string s);

  /// Puts a result at the end and get a count of the number of arguments actually added
  Option *add_result(std::string s, int &results_added);

  /// Puts a result at the end
  Option *add_result(std::vector<std::string> s);

  /// Get the current complete results set
  CLI11_NODISCARD const results_t &results() const { return results_; }

  /// Get a copy of the results
  CLI11_NODISCARD results_t reduced_results() const;

  /// Get the results as a specified type
  template <typename T>
  void results(T &output) const {
    bool retval = false;
    if (current_option_state_ >= option_state::reduced || (results_.size() == 1 && validators_.empty())) {
      const results_t &res = (proc_results_.empty()) ? results_ : proc_results_;
      retval = detail::lexical_conversion<T, T>(res, output);
    } else {
      results_t res;
      if (results_.empty()) {
        if (!default_str_.empty()) {
          // _add_results takes an rvalue only
          _add_result(std::string(default_str_), res);
          _validate_results(res);
          results_t extra;
          _reduce_results(extra, res);
          if (!extra.empty()) {
            res = std::move(extra);
          }
        } else {
          res.emplace_back();
        }
      } else {
        res = reduced_results();
      }
      retval = detail::lexical_conversion<T, T>(res, output);
    }
    if (!retval) {
      throw ConversionError(get_name(), results_);
    }
  }

  /// Return the results as the specified type
  template <typename T>
  CLI11_NODISCARD T as() const {
    T output;
    results(output);
    return output;
  }

  /// See if the callback has been run already
  CLI11_NODISCARD bool get_callback_run() const { return (current_option_state_ == option_state::callback_run); }

  ///@}
  /// @name Custom options
  ///@{

  /// Set the type function to run when displayed on this option
  Option *type_name_fn(std::function<std::string()> typefun) {
    type_name_ = std::move(typefun);
    return this;
  }

  /// Set a custom option typestring
  Option *type_name(std::string typeval) {
    type_name_fn([typeval]() { return typeval; });
    return this;
  }

  /// Set a custom option size
  Option *type_size(int option_type_size);

  /// Set a custom option type size range
  Option *type_size(int option_type_size_min, int option_type_size_max);

  /// Set the value of the separator injection flag
  void inject_separator(bool value = true) { inject_separator_ = value; }

  /// Set a capture function for the default. Mostly used by App.
  Option *default_function(const std::function<std::string()> &func) {
    default_function_ = func;
    return this;
  }

  /// Capture the default value from the original value (if it can be captured)
  Option *capture_default_str() {
    if (default_function_) {
      default_str_ = default_function_();
    }
    return this;
  }

  /// Set the default value string representation (does not change the contained value)
  Option *default_str(std::string val) {
    default_str_ = std::move(val);
    return this;
  }

  /// Set the default value and validate the results and run the callback if appropriate to set the value into the
  /// bound value only available for types that can be converted to a string
  template <typename X>
  Option *default_val(const X &val) {
    std::string val_str = detail::to_string(val);
    auto old_option_state = current_option_state_;
    results_t old_results{std::move(results_)};
    results_.clear();
    try {
      add_result(val_str);
      // if trigger_on_result_ is set the callback already ran
      if (run_callback_for_default_ && !trigger_on_result_) {
        run_callback(); // run callback sets the state, we need to reset it again
        current_option_state_ = option_state::parsing;
      } else {
        _validate_results(results_);
        current_option_state_ = old_option_state;
      }
    } catch (const CLI::Error &) {
      // this should be done
      results_ = std::move(old_results);
      current_option_state_ = old_option_state;
      throw;
    }
    results_ = std::move(old_results);
    default_str_ = std::move(val_str);
    return this;
  }

  /// Get the full typename for this option
  CLI11_NODISCARD std::string get_type_name() const;

private:
  /// Run the results through the Validators
  void _validate_results(results_t &res) const;

  /** reduce the results in accordance with the MultiOptionPolicy
  @param[out] out results are assigned to res if there if they are different
  */
  void _reduce_results(results_t &out, const results_t &original) const;

  // Run a result through the Validators
  std::string _validate(std::string &result, int index) const;

  /// Add a single result to the result set, taking into account delimiters
  int _add_result(std::string &&result, std::vector<std::string> &res) const;
};

template <typename CRTP>
template <typename T>
void OptionBase<CRTP>::copy_to(T *other) const {
  other->group(group_);
  other->required(required_);
  other->ignore_case(ignore_case_);
  other->ignore_underscore(ignore_underscore_);
  other->configurable(configurable_);
  other->disable_flag_override(disable_flag_override_);
  other->delimiter(delimiter_);
  other->always_capture_default(always_capture_default_);
  other->multi_option_policy(multi_option_policy_);
}

CLI11_INLINE Option *Option::expected(int value) {
  if (value < 0) {
    expected_min_ = -value;
    if (expected_max_ < expected_min_) {
      expected_max_ = expected_min_;
    }
    allow_extra_args_ = true;
    flag_like_ = false;
  } else if (value == detail::expected_max_vector_size) {
    expected_min_ = 1;
    expected_max_ = detail::expected_max_vector_size;
    allow_extra_args_ = true;
    flag_like_ = false;
  } else {
    expected_min_ = value;
    expected_max_ = value;
    flag_like_ = (expected_min_ == 0);
  }
  return this;
}

CLI11_INLINE Option *Option::expected(int value_min, int value_max) {
  if (value_min < 0) {
    value_min = -value_min;
  }

  if (value_max < 0) {
    value_max = detail::expected_max_vector_size;
  }
  if (value_max < value_min) {
    expected_min_ = value_max;
    expected_max_ = value_min;
  } else {
    expected_max_ = value_max;
    expected_min_ = value_min;
  }

  return this;
}

CLI11_INLINE Option *Option::check(Validator validator, const std::string &validator_name) {
  validator.non_modifying();
  validators_.push_back(std::move(validator));
  if (!validator_name.empty())
    validators_.back().name(validator_name);
  return this;
}

CLI11_INLINE Option *Option::check(std::function<std::string(const std::string &)> Validator,
                                   std::string Validator_description,
                                   std::string Validator_name) {
  validators_.emplace_back(Validator, std::move(Validator_description), std::move(Validator_name));
  validators_.back().non_modifying();
  return this;
}

CLI11_INLINE Option *Option::transform(Validator Validator, const std::string &Validator_name) {
  validators_.insert(validators_.begin(), std::move(Validator));
  if (!Validator_name.empty())
    validators_.front().name(Validator_name);
  return this;
}

CLI11_INLINE Option *Option::transform(const std::function<std::string(std::string)> &func,
                                       std::string transform_description,
                                       std::string transform_name) {
  validators_.insert(validators_.begin(),
                     Validator(
                         [func](std::string &val) {
                           val = func(val);
                           return std::string{};
                         },
                         std::move(transform_description),
                         std::move(transform_name)));

  return this;
}

CLI11_INLINE Option *Option::each(const std::function<void(std::string)> &func) {
  validators_.emplace_back(
      [func](std::string &inout) {
        func(inout);
        return std::string{};
      },
      std::string{});
  return this;
}

CLI11_INLINE Validator *Option::get_validator(const std::string &Validator_name) {
  for (auto &Validator : validators_) {
    if (Validator_name == Validator.get_name()) {
      return &Validator;
    }
  }
  if ((Validator_name.empty()) && (!validators_.empty())) {
    return &(validators_.front());
  }
  throw OptionNotFound(std::string{"Validator "} + Validator_name + " Not Found");
}

CLI11_INLINE Validator *Option::get_validator(int index) {
  // This is an signed int so that it is not equivalent to a pointer.
  if (index >= 0 && index < static_cast<int>(validators_.size())) {
    return &(validators_[static_cast<decltype(validators_)::size_type>(index)]);
  }
  throw OptionNotFound("Validator index is not valid");
}

CLI11_INLINE bool Option::remove_needs(Option *opt) {
  auto iterator = std::find(std::begin(needs_), std::end(needs_), opt);

  if (iterator == std::end(needs_)) {
    return false;
  }
  needs_.erase(iterator);
  return true;
}

CLI11_INLINE Option *Option::excludes(Option *opt) {
  if (opt == this) {
    throw(IncorrectConstruction("and option cannot exclude itself"));
  }
  excludes_.insert(opt);

  // Help text should be symmetric - excluding a should exclude b
  opt->excludes_.insert(this);

  // Ignoring the insert return value, excluding twice is now allowed.
  // (Mostly to allow both directions to be excluded by user, even though the library does it for you.)

  return this;
}

CLI11_INLINE bool Option::remove_excludes(Option *opt) {
  auto iterator = std::find(std::begin(excludes_), std::end(excludes_), opt);

  if (iterator == std::end(excludes_)) {
    return false;
  }
  excludes_.erase(iterator);
  return true;
}

template <typename T>
Option *Option::ignore_case(bool value) {
  if (!ignore_case_ && value) {
    ignore_case_ = value;
    auto *parent = static_cast<T *>(parent_);
    for (const Option_p &opt : parent->options_) {
      if (opt.get() == this) {
        continue;
      }
      const auto &omatch = opt->matching_name(*this);
      if (!omatch.empty()) {
        ignore_case_ = false;
        throw OptionAlreadyAdded("adding ignore case caused a name conflict with " + omatch);
      }
    }
  } else {
    ignore_case_ = value;
  }
  return this;
}

template <typename T>
Option *Option::ignore_underscore(bool value) {

  if (!ignore_underscore_ && value) {
    ignore_underscore_ = value;
    auto *parent = static_cast<T *>(parent_);
    for (const Option_p &opt : parent->options_) {
      if (opt.get() == this) {
        continue;
      }
      const auto &omatch = opt->matching_name(*this);
      if (!omatch.empty()) {
        ignore_underscore_ = false;
        throw OptionAlreadyAdded("adding ignore underscore caused a name conflict with " + omatch);
      }
    }
  } else {
    ignore_underscore_ = value;
  }
  return this;
}

CLI11_INLINE Option *Option::multi_option_policy(MultiOptionPolicy value) {
  if (value != multi_option_policy_) {
    if (multi_option_policy_ == MultiOptionPolicy::Throw && expected_max_ == detail::expected_max_vector_size &&
        expected_min_ > 1) { // this bizarre condition is to maintain backwards compatibility
                             // with the previous behavior of expected_ with vectors
      expected_max_ = expected_min_;
    }
    multi_option_policy_ = value;
    current_option_state_ = option_state::parsing;
  }
  return this;
}

CLI11_NODISCARD CLI11_INLINE std::string Option::get_name(bool positional, bool all_options) const {
  if (get_group().empty())
    return {}; // Hidden

  if (all_options) {

    std::vector<std::string> name_list;

    /// The all list will never include a positional unless asked or that's the only name.
    if ((positional && (!pname_.empty())) || (snames_.empty() && lnames_.empty())) {
      name_list.push_back(pname_);
    }
    if ((get_items_expected() == 0) && (!fnames_.empty())) {
      for (const std::string &sname : snames_) {
        name_list.push_back("-" + sname);
        if (check_fname(sname)) {
          name_list.back() += "{" + get_flag_value(sname, "") + "}";
        }
      }

      for (const std::string &lname : lnames_) {
        name_list.push_back("--" + lname);
        if (check_fname(lname)) {
          name_list.back() += "{" + get_flag_value(lname, "") + "}";
        }
      }
    } else {
      for (const std::string &sname : snames_)
        name_list.push_back("-" + sname);

      for (const std::string &lname : lnames_)
        name_list.push_back("--" + lname);
    }

    return detail::join(name_list);
  }

  // This returns the positional name no matter what
  if (positional)
    return pname_;

  // Prefer long name
  if (!lnames_.empty())
    return std::string(2, '-') + lnames_[0];

  // Or short name if no long name
  if (!snames_.empty())
    return std::string(1, '-') + snames_[0];

  // If positional is the only name, it's okay to use that
  return pname_;
}

CLI11_INLINE void Option::run_callback() {
  if (force_callback_ && results_.empty()) {
    add_result(default_str_);
  }
  if (current_option_state_ == option_state::parsing) {
    _validate_results(results_);
    current_option_state_ = option_state::validated;
  }

  if (current_option_state_ < option_state::reduced) {
    _reduce_results(proc_results_, results_);
    current_option_state_ = option_state::reduced;
  }
  if (current_option_state_ >= option_state::reduced) {
    current_option_state_ = option_state::callback_run;
    if (!(callback_)) {
      return;
    }
    const results_t &send_results = proc_results_.empty() ? results_ : proc_results_;
    bool local_result = callback_(send_results);

    if (!local_result)
      throw ConversionError(get_name(), results_);
  }
}

CLI11_NODISCARD CLI11_INLINE const std::string &Option::matching_name(const Option &other) const {
  static const std::string estring;
  for (const std::string &sname : snames_) {
    if (other.check_sname(sname))
      return sname;
    if (other.check_lname(sname))
      return sname;
  }
  for (const std::string &lname : lnames_) {
    if (other.check_lname(lname))
      return lname;
    if (lname.size() == 1) {
      if (other.check_sname(lname)) {
        return lname;
      }
    }
  }
  if (snames_.empty() && lnames_.empty() && !pname_.empty()) {
    if (other.check_sname(pname_) || other.check_lname(pname_) || pname_ == other.pname_)
      return pname_;
  }
  if (other.snames_.empty() && other.fnames_.empty() && !other.pname_.empty()) {
    if (check_sname(other.pname_) || check_lname(other.pname_) || (pname_ == other.pname_))
      return other.pname_;
  }
  if (ignore_case_ ||
      ignore_underscore_) { // We need to do the inverse, in case we are ignore_case or ignore underscore
    for (const std::string &sname : other.snames_)
      if (check_sname(sname))
        return sname;
    for (const std::string &lname : other.lnames_)
      if (check_lname(lname))
        return lname;
  }
  return estring;
}

CLI11_NODISCARD CLI11_INLINE bool Option::check_name(const std::string &name) const {

  if (name.length() > 2 && name[0] == '-' && name[1] == '-')
    return check_lname(name.substr(2));
  if (name.length() > 1 && name.front() == '-')
    return check_sname(name.substr(1));
  if (!pname_.empty()) {
    std::string local_pname = pname_;
    std::string local_name = name;
    if (ignore_underscore_) {
      local_pname = detail::remove_underscore(local_pname);
      local_name = detail::remove_underscore(local_name);
    }
    if (ignore_case_) {
      local_pname = detail::to_lower(local_pname);
      local_name = detail::to_lower(local_name);
    }
    if (local_name == local_pname) {
      return true;
    }
  }

  if (!envname_.empty()) {
    // this needs to be the original since envname_ shouldn't match on case insensitivity
    return (name == envname_);
  }
  return false;
}

CLI11_NODISCARD CLI11_INLINE std::string Option::get_flag_value(const std::string &name,
                                                                std::string input_value) const {
  static const std::string trueString{"true"};
  static const std::string falseString{"false"};
  static const std::string emptyString{"{}"};
  // check for disable flag override_
  if (disable_flag_override_) {
    if (!((input_value.empty()) || (input_value == emptyString))) {
      auto default_ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
      if (default_ind >= 0) {
        // We can static cast this to std::size_t because it is more than 0 in this block
        if (default_flag_values_[static_cast<std::size_t>(default_ind)].second != input_value) {
          if (input_value == default_str_ && force_callback_) {
            return input_value;
          }
          throw(ArgumentMismatch::FlagOverride(name));
        }
      } else {
        if (input_value != trueString) {
          throw(ArgumentMismatch::FlagOverride(name));
        }
      }
    }
  }
  auto ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
  if ((input_value.empty()) || (input_value == emptyString)) {
    if (flag_like_) {
      return (ind < 0) ? trueString : default_flag_values_[static_cast<std::size_t>(ind)].second;
    }
    return (ind < 0) ? default_str_ : default_flag_values_[static_cast<std::size_t>(ind)].second;
  }
  if (ind < 0) {
    return input_value;
  }
  if (default_flag_values_[static_cast<std::size_t>(ind)].second == falseString) {
    errno = 0;
    auto val = detail::to_flag_value(input_value);
    if (errno != 0) {
      errno = 0;
      return input_value;
    }
    return (val == 1) ? falseString : (val == (-1) ? trueString : std::to_string(-val));
  }
  return input_value;
}

CLI11_INLINE Option *Option::add_result(std::string s) {
  _add_result(std::move(s), results_);
  current_option_state_ = option_state::parsing;
  return this;
}

CLI11_INLINE Option *Option::add_result(std::string s, int &results_added) {
  results_added = _add_result(std::move(s), results_);
  current_option_state_ = option_state::parsing;
  return this;
}

CLI11_INLINE Option *Option::add_result(std::vector<std::string> s) {
  current_option_state_ = option_state::parsing;
  for (auto &str : s) {
    _add_result(std::move(str), results_);
  }
  return this;
}

CLI11_NODISCARD CLI11_INLINE results_t Option::reduced_results() const {
  results_t res = proc_results_.empty() ? results_ : proc_results_;
  if (current_option_state_ < option_state::reduced) {
    if (current_option_state_ == option_state::parsing) {
      res = results_;
      _validate_results(res);
    }
    if (!res.empty()) {
      results_t extra;
      _reduce_results(extra, res);
      if (!extra.empty()) {
        res = std::move(extra);
      }
    }
  }
  return res;
}

CLI11_INLINE Option *Option::type_size(int option_type_size) {
  if (option_type_size < 0) {
    // this section is included for backwards compatibility
    type_size_max_ = -option_type_size;
    type_size_min_ = -option_type_size;
    expected_max_ = detail::expected_max_vector_size;
  } else {
    type_size_max_ = option_type_size;
    if (type_size_max_ < detail::expected_max_vector_size) {
      type_size_min_ = option_type_size;
    } else {
      inject_separator_ = true;
    }
    if (type_size_max_ == 0)
      required_ = false;
  }
  return this;
}

CLI11_INLINE Option *Option::type_size(int option_type_size_min, int option_type_size_max) {
  if (option_type_size_min < 0 || option_type_size_max < 0) {
    // this section is included for backwards compatibility
    expected_max_ = detail::expected_max_vector_size;
    option_type_size_min = (std::abs)(option_type_size_min);
    option_type_size_max = (std::abs)(option_type_size_max);
  }

  if (option_type_size_min > option_type_size_max) {
    type_size_max_ = option_type_size_min;
    type_size_min_ = option_type_size_max;
  } else {
    type_size_min_ = option_type_size_min;
    type_size_max_ = option_type_size_max;
  }
  if (type_size_max_ == 0) {
    required_ = false;
  }
  if (type_size_max_ >= detail::expected_max_vector_size) {
    inject_separator_ = true;
  }
  return this;
}

CLI11_NODISCARD CLI11_INLINE std::string Option::get_type_name() const {
  std::string full_type_name = type_name_();
  if (!validators_.empty()) {
    for (const auto &Validator : validators_) {
      std::string vtype = Validator.get_description();
      if (!vtype.empty()) {
        full_type_name += ":" + vtype;
      }
    }
  }
  return full_type_name;
}

CLI11_INLINE void Option::_validate_results(results_t &res) const {
  // Run the Validators (can change the string)
  if (!validators_.empty()) {
    if (type_size_max_ > 1) { // in this context index refers to the index in the type
      int index = 0;
      if (get_items_expected_max() < static_cast<int>(res.size()) &&
          (multi_option_policy_ == CLI::MultiOptionPolicy::TakeLast ||
           multi_option_policy_ == CLI::MultiOptionPolicy::Reverse)) {
        // create a negative index for the earliest ones
        index = get_items_expected_max() - static_cast<int>(res.size());
      }

      for (std::string &result : res) {
        if (detail::is_separator(result) && type_size_max_ != type_size_min_ && index >= 0) {
          index = 0; // reset index for variable size chunks
          continue;
        }
        auto err_msg = _validate(result, (index >= 0) ? (index % type_size_max_) : index);
        if (!err_msg.empty())
          throw ValidationError(get_name(), err_msg);
        ++index;
      }
    } else {
      int index = 0;
      if (expected_max_ < static_cast<int>(res.size()) &&
          (multi_option_policy_ == CLI::MultiOptionPolicy::TakeLast ||
           multi_option_policy_ == CLI::MultiOptionPolicy::Reverse)) {
        // create a negative index for the earliest ones
        index = expected_max_ - static_cast<int>(res.size());
      }
      for (std::string &result : res) {
        auto err_msg = _validate(result, index);
        ++index;
        if (!err_msg.empty())
          throw ValidationError(get_name(), err_msg);
      }
    }
  }
}

CLI11_INLINE void Option::_reduce_results(results_t &out, const results_t &original) const {

  // max num items expected or length of vector, always at least 1
  // Only valid for a trimming policy

  out.clear();
  // Operation depends on the policy setting
  switch (multi_option_policy_) {
  case MultiOptionPolicy::TakeAll:
    break;
  case MultiOptionPolicy::TakeLast: {
    // Allow multi-option sizes (including 0)
    std::size_t trim_size = std::min<std::size_t>(
        static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
    if (original.size() != trim_size) {
      out.assign(original.end() - static_cast<results_t::difference_type>(trim_size), original.end());
    }
  } break;
  case MultiOptionPolicy::Reverse: {
    // Allow multi-option sizes (including 0)
    std::size_t trim_size = std::min<std::size_t>(
        static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
    if (original.size() != trim_size || trim_size > 1) {
      out.assign(original.end() - static_cast<results_t::difference_type>(trim_size), original.end());
    }
    std::reverse(out.begin(), out.end());
  } break;
  case MultiOptionPolicy::TakeFirst: {
    std::size_t trim_size = std::min<std::size_t>(
        static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
    if (original.size() != trim_size) {
      out.assign(original.begin(), original.begin() + static_cast<results_t::difference_type>(trim_size));
    }
  } break;
  case MultiOptionPolicy::Join:
    if (results_.size() > 1) {
      out.push_back(detail::join(original, std::string(1, (delimiter_ == '\0') ? '\n' : delimiter_)));
    }
    break;
  case MultiOptionPolicy::Sum:
    out.push_back(detail::sum_string_vector(original));
    break;
  case MultiOptionPolicy::Throw:
  default: {
    auto num_min = static_cast<std::size_t>(get_items_expected_min());
    auto num_max = static_cast<std::size_t>(get_items_expected_max());
    if (num_min == 0) {
      num_min = 1;
    }
    if (num_max == 0) {
      num_max = 1;
    }
    if (original.size() < num_min) {
      throw ArgumentMismatch::AtLeast(get_name(), static_cast<int>(num_min), original.size());
    }
    if (original.size() > num_max) {
      if (original.size() == 2 && num_max == 1 && original[1] == "%%" && original[0] == "{}") {
        // this condition is a trap for the following empty indicator check on config files
        out = original;
      } else {
        throw ArgumentMismatch::AtMost(get_name(), static_cast<int>(num_max), original.size());
      }
    }
    break;
  }
  }
  // this check is to allow an empty vector in certain circumstances but not if expected is not zero.
  // {} is the indicator for an empty container
  if (out.empty()) {
    if (original.size() == 1 && original[0] == "{}" && get_items_expected_min() > 0) {
      out.emplace_back("{}");
      out.emplace_back("%%");
    }
  } else if (out.size() == 1 && out[0] == "{}" && get_items_expected_min() > 0) {
    out.emplace_back("%%");
  }
}

CLI11_INLINE std::string Option::_validate(std::string &result, int index) const {
  std::string err_msg;
  if (result.empty() && expected_min_ == 0) {
    // an empty with nothing expected is allowed
    return err_msg;
  }
  for (const auto &vali : validators_) {
    auto v = vali.get_application_index();
    if (v == -1 || v == index) {
      try {
        err_msg = vali(result);
      } catch (const ValidationError &err) {
        err_msg = err.what();
      }
      if (!err_msg.empty())
        break;
    }
  }

  return err_msg;
}

CLI11_INLINE int Option::_add_result(std::string &&result, std::vector<std::string> &res) const {
  int result_count = 0;
  if (allow_extra_args_ && !result.empty() && result.front() == '[' &&
      result.back() == ']') { // this is now a vector string likely from the default or user entry
    result.pop_back();

    for (auto &var : CLI::detail::split(result.substr(1), ',')) {
      if (!var.empty()) {
        result_count += _add_result(std::move(var), res);
      }
    }
    return result_count;
  }
  if (delimiter_ == '\0') {
    res.push_back(std::move(result));
    ++result_count;
  } else {
    if ((result.find_first_of(delimiter_) != std::string::npos)) {
      for (const auto &var : CLI::detail::split(result, delimiter_)) {
        if (!var.empty()) {
          res.push_back(var);
          ++result_count;
        }
      }
    } else {
      res.push_back(std::move(result));
      ++result_count;
    }
  }
  return result_count;
}

#ifndef CLI11_PARSE
#define CLI11_PARSE(app, ...)          \
  try {                                \
    (app).parse(__VA_ARGS__);          \
  } catch (const CLI::ParseError &e) { \
    return (app).exit(e);              \
  }
#endif

namespace detail {
enum class Classifier { NONE,
                        POSITIONAL_MARK,
                        SHORT,
                        LONG,
                        WINDOWS_STYLE,
                        SUBCOMMAND,
                        SUBCOMMAND_TERMINATOR };
struct AppFriend;
} // namespace detail

namespace FailureMessage {
/// Printout a clean, simple message on error (the default in CLI11 1.5+)
CLI11_INLINE std::string simple(const App *app, const Error &e);

/// Printout the full help string on error (if this fn is set, the old default for CLI11)
CLI11_INLINE std::string help(const App *app, const Error &e);
} // namespace FailureMessage

/// enumeration of modes of how to deal with extras in config files

enum class config_extras_mode : char { error = 0,
                                       ignore,
                                       ignore_all,
                                       capture };

class App;

using App_p = std::shared_ptr<App>;

namespace detail {
/// helper functions for adding in appropriate flag modifiers for add_flag

template <typename T, enable_if_t<!std::is_integral<T>::value || (sizeof(T) <= 1U), detail::enabler> = detail::dummy>
Option *default_flag_modifiers(Option *opt) {
  return opt->always_capture_default();
}

/// summing modifiers
template <typename T, enable_if_t<std::is_integral<T>::value && (sizeof(T) > 1U), detail::enabler> = detail::dummy>
Option *default_flag_modifiers(Option *opt) {
  return opt->multi_option_policy(MultiOptionPolicy::Sum)->default_str("0")->force_callback();
}

} // namespace detail

class Option_group;
/// Creates a command line program, with very few defaults.
/** To use, create a new `Program()` instance with `argc`, `argv`, and a help description. The templated
 *  add_option methods make it easy to prepare options. Remember to call `.start` before starting your
 * program, so that the options can be evaluated and the help option doesn't accidentally run your program. */
class App {
  friend Option;
  friend detail::AppFriend;

protected:
  // This library follows the Google style guide for member names ending in underscores

  /// @name Basics
  ///@{

  /// Subcommand name or program name (from parser if name is empty)
  std::string name_{};

  /// Description of the current program/subcommand
  std::string description_{};

  /// If true, allow extra arguments (ie, don't throw an error). INHERITABLE
  bool allow_extras_{false};

  /// If ignore, allow extra arguments in the ini file (ie, don't throw an error). INHERITABLE
  /// if error error on an extra argument, and if capture feed it to the app
  config_extras_mode allow_config_extras_{config_extras_mode::ignore};

  ///  If true, return immediately on an unrecognized option (implies allow_extras) INHERITABLE
  bool prefix_command_{false};

  /// If set to true the name was automatically generated from the command line vs a user set name
  bool has_automatic_name_{false};

  /// If set to true the subcommand is required to be processed and used, ignored for main app
  bool required_{false};

  /// If set to true the subcommand is disabled and cannot be used, ignored for main app
  bool disabled_{false};

  /// Flag indicating that the pre_parse_callback has been triggered
  bool pre_parse_called_{false};

  /// Flag indicating that the callback for the subcommand should be executed immediately on parse completion which is
  /// before help or ini files are processed. INHERITABLE
  bool immediate_callback_{false};

  /// This is a function that runs prior to the start of parsing
  std::function<void(std::size_t)> pre_parse_callback_{};

  /// This is a function that runs when parsing has finished.
  std::function<void()> parse_complete_callback_{};

  /// This is a function that runs when all processing has completed
  std::function<void()> final_callback_{};

  ///@}
  /// @name Options
  ///@{

  /// The default values for options, customizable and changeable INHERITABLE
  OptionDefaults option_defaults_{};

  /// The list of options, stored locally
  std::vector<Option_p> options_{};

  ///@}
  /// @name Help
  ///@{

  /// Usage to put after program/subcommand description in the help output INHERITABLE
  std::string usage_{};

  /// This is a function that generates a usage to put after program/subcommand description in help output
  std::function<std::string()> usage_callback_{};

  /// Footer to put after all options in the help output INHERITABLE
  std::string footer_{};

  /// This is a function that generates a footer to put after all other options in help output
  std::function<std::string()> footer_callback_{};

  /// A pointer to the help flag if there is one INHERITABLE
  Option *help_ptr_{nullptr};

  /// A pointer to the help all flag if there is one INHERITABLE
  Option *help_all_ptr_{nullptr};

  /// A pointer to a version flag if there is one
  Option *version_ptr_{nullptr};

  /// This is the formatter for help printing. Default provided. INHERITABLE (same pointer)
  std::shared_ptr<FormatterBase> formatter_{new Formatter()};

  /// The error message printing function INHERITABLE
  std::function<std::string(const App *, const Error &e)> failure_message_{FailureMessage::simple};

  ///@}
  /// @name Parsing
  ///@{

  using missing_t = std::vector<std::pair<detail::Classifier, std::string>>;

  /// Pair of classifier, string for missing options. (extra detail is removed on returning from parse)
  ///
  /// This is faster and cleaner than storing just a list of strings and reparsing. This may contain the -- separator.
  missing_t missing_{};

  /// This is a list of pointers to options with the original parse order
  std::vector<Option *> parse_order_{};

  /// This is a list of the subcommands collected, in order
  std::vector<App *> parsed_subcommands_{};

  /// this is a list of subcommands that are exclusionary to this one
  std::set<App *> exclude_subcommands_{};

  /// This is a list of options which are exclusionary to this App, if the options were used this subcommand should
  /// not be
  std::set<Option *> exclude_options_{};

  /// this is a list of subcommands or option groups that are required by this one, the list is not mutual,  the
  /// listed subcommands do not require this one
  std::set<App *> need_subcommands_{};

  /// This is a list of options which are required by this app, the list is not mutual, listed options do not need the
  /// subcommand not be
  std::set<Option *> need_options_{};

  ///@}
  /// @name Subcommands
  ///@{

  /// Storage for subcommand list
  std::vector<App_p> subcommands_{};

  /// If true, the program name is not case sensitive INHERITABLE
  bool ignore_case_{false};

  /// If true, the program should ignore underscores INHERITABLE
  bool ignore_underscore_{false};

  /// Allow subcommand fallthrough, so that parent commands can collect commands after subcommand.  INHERITABLE
  bool fallthrough_{false};

  /// Allow '/' for options for Windows like options. Defaults to true on Windows, false otherwise. INHERITABLE
  bool allow_windows_style_options_{
#ifdef _WIN32
      true
#else
      false
#endif
  };
  /// specify that positional arguments come at the end of the argument sequence not inheritable
  bool positionals_at_end_{false};

  enum class startup_mode : char { stable,
                                   enabled,
                                   disabled };
  /// specify the startup mode for the app
  /// stable=no change, enabled= startup enabled, disabled=startup disabled
  startup_mode default_startup{startup_mode::stable};

  /// if set to true the subcommand can be triggered via configuration files INHERITABLE
  bool configurable_{false};

  /// If set to true positional options are validated before assigning INHERITABLE
  bool validate_positionals_{false};

  /// If set to true optional vector arguments are validated before assigning INHERITABLE
  bool validate_optional_arguments_{false};

  /// indicator that the subcommand is silent and won't show up in subcommands list
  /// This is potentially useful as a modifier subcommand
  bool silent_{false};

  /// Counts the number of times this command/subcommand was parsed
  std::uint32_t parsed_{0U};

  /// Minimum required subcommands (not inheritable!)
  std::size_t require_subcommand_min_{0};

  /// Max number of subcommands allowed (parsing stops after this number). 0 is unlimited INHERITABLE
  std::size_t require_subcommand_max_{0};

  /// Minimum required options (not inheritable!)
  std::size_t require_option_min_{0};

  /// Max number of options allowed. 0 is unlimited (not inheritable)
  std::size_t require_option_max_{0};

  /// A pointer to the parent if this is a subcommand
  App *parent_{nullptr};

  /// The group membership INHERITABLE
  std::string group_{"Subcommands"};

  /// Alias names for the subcommand
  std::vector<std::string> aliases_{};

  ///@}
  /// @name Config
  ///@{

  /// Pointer to the config option
  Option *config_ptr_{nullptr};

  /// This is the formatter for help printing. Default provided. INHERITABLE (same pointer)
  std::shared_ptr<Config> config_formatter_{new ConfigTOML()};

  ///@}

#ifdef _WIN32
  /// When normalizing argv to UTF-8 on Windows, this is the storage for normalized args.
  std::vector<std::string> normalized_argv_{};

  /// When normalizing argv to UTF-8 on Windows, this is the `char**` value returned to the user.
  std::vector<char *> normalized_argv_view_{};
#endif

  /// Special private constructor for subcommand
  App(std::string app_description, std::string app_name, App *parent);

public:
  /// @name Basic
  ///@{

  /// Create a new program. Pass in the same arguments as main(), along with a help string.
  explicit App(std::string app_description = "", std::string app_name = "")
      : App(app_description, app_name, nullptr) {
    set_help_flag("-h,--help", "Print this help message and exit");
  }

  App(const App &) = delete;
  App &operator=(const App &) = delete;

  /// virtual destructor
  virtual ~App() = default;

  /// Convert the contents of argv to UTF-8. Only does something on Windows, does nothing elsewhere.
  CLI11_NODISCARD char **ensure_utf8(char **argv);

  /// Set a callback for execution when all parsing and processing has completed
  ///
  /// Due to a bug in c++11,
  /// it is not possible to overload on std::function (fixed in c++14
  /// and backported to c++11 on newer compilers). Use capture by reference
  /// to get a pointer to App if needed.
  App *callback(std::function<void()> app_callback) {
    if (immediate_callback_) {
      parse_complete_callback_ = std::move(app_callback);
    } else {
      final_callback_ = std::move(app_callback);
    }
    return this;
  }

  /// Set a callback for execution when all parsing and processing has completed
  /// aliased as callback
  App *final_callback(std::function<void()> app_callback) {
    final_callback_ = std::move(app_callback);
    return this;
  }

  /// Set a callback to execute when parsing has completed for the app
  ///
  App *parse_complete_callback(std::function<void()> pc_callback) {
    parse_complete_callback_ = std::move(pc_callback);
    return this;
  }

  /// Set a callback to execute prior to parsing.
  ///
  App *preparse_callback(std::function<void(std::size_t)> pp_callback) {
    pre_parse_callback_ = std::move(pp_callback);
    return this;
  }

  /// Set a name for the app (empty will use parser to set the name)
  App *name(std::string app_name = "");

  /// Set an alias for the app
  App *alias(std::string app_name);

  /// Remove the error when extras are left over on the command line.
  App *allow_extras(bool allow = true) {
    allow_extras_ = allow;
    return this;
  }

  /// Remove the error when extras are left over on the command line.
  App *required(bool require = true) {
    required_ = require;
    return this;
  }

  /// Disable the subcommand or option group
  App *disabled(bool disable = true) {
    disabled_ = disable;
    return this;
  }

  /// silence the subcommand from showing up in the processed list
  App *silent(bool silence = true) {
    silent_ = silence;
    return this;
  }

  /// Set the subcommand to be disabled by default, so on clear(), at the start of each parse it is disabled
  App *disabled_by_default(bool disable = true) {
    if (disable) {
      default_startup = startup_mode::disabled;
    } else {
      default_startup = (default_startup == startup_mode::enabled) ? startup_mode::enabled : startup_mode::stable;
    }
    return this;
  }

  /// Set the subcommand to be enabled by default, so on clear(), at the start of each parse it is enabled (not
  /// disabled)
  App *enabled_by_default(bool enable = true) {
    if (enable) {
      default_startup = startup_mode::enabled;
    } else {
      default_startup =
          (default_startup == startup_mode::disabled) ? startup_mode::disabled : startup_mode::stable;
    }
    return this;
  }

  /// Set the subcommand callback to be executed immediately on subcommand completion
  App *immediate_callback(bool immediate = true);

  /// Set the subcommand to validate positional arguments before assigning
  App *validate_positionals(bool validate = true) {
    validate_positionals_ = validate;
    return this;
  }

  /// Set the subcommand to validate optional vector arguments before assigning
  App *validate_optional_arguments(bool validate = true) {
    validate_optional_arguments_ = validate;
    return this;
  }

  /// ignore extras in config files
  App *allow_config_extras(bool allow = true) {
    if (allow) {
      allow_config_extras_ = config_extras_mode::capture;
      allow_extras_ = true;
    } else {
      allow_config_extras_ = config_extras_mode::error;
    }
    return this;
  }

  /// ignore extras in config files
  App *allow_config_extras(config_extras_mode mode) {
    allow_config_extras_ = mode;
    return this;
  }

  /// Do not parse anything after the first unrecognized option and return
  App *prefix_command(bool allow = true) {
    prefix_command_ = allow;
    return this;
  }

  /// Ignore case. Subcommands inherit value.
  App *ignore_case(bool value = true);

  /// Allow windows style options, such as `/opt`. First matching short or long name used. Subcommands inherit
  /// value.
  App *allow_windows_style_options(bool value = true) {
    allow_windows_style_options_ = value;
    return this;
  }

  /// Specify that the positional arguments are only at the end of the sequence
  App *positionals_at_end(bool value = true) {
    positionals_at_end_ = value;
    return this;
  }

  /// Specify that the subcommand can be triggered by a config file
  App *configurable(bool value = true) {
    configurable_ = value;
    return this;
  }

  /// Ignore underscore. Subcommands inherit value.
  App *ignore_underscore(bool value = true);

  /// Set the help formatter
  App *formatter(std::shared_ptr<FormatterBase> fmt) {
    formatter_ = fmt;
    return this;
  }

  /// Set the help formatter
  App *formatter_fn(std::function<std::string(const App *, std::string, AppFormatMode)> fmt) {
    formatter_ = std::make_shared<FormatterLambda>(fmt);
    return this;
  }

  /// Set the config formatter
  App *config_formatter(std::shared_ptr<Config> fmt) {
    config_formatter_ = fmt;
    return this;
  }

  /// Check to see if this subcommand was parsed, true only if received on command line.
  CLI11_NODISCARD bool parsed() const { return parsed_ > 0; }

  /// Get the OptionDefault object, to set option defaults
  OptionDefaults *option_defaults() { return &option_defaults_; }

  ///@}
  /// @name Adding options
  ///@{

  /// Add an option, will automatically understand the type for common types.
  ///
  /// To use, create a variable with the expected type, and pass it in after the name.
  /// After start is called, you can use count to see if the value was passed, and
  /// the value will be initialized properly. Numbers, vectors, and strings are supported.
  ///
  /// ->required(), ->default, and the validators are options,
  /// The positional options take an optional number of arguments.
  ///
  /// For example,
  ///
  ///     std::string filename;
  ///     program.add_option("filename", filename, "description of filename");
  ///
  Option *add_option(std::string option_name,
                     callback_t option_callback,
                     std::string option_description = "",
                     bool defaulted = false,
                     std::function<std::string()> func = {});

  /// Add option for assigning to a variable
  template <typename AssignTo,
            typename ConvertTo = AssignTo,
            enable_if_t<!std::is_const<ConvertTo>::value, detail::enabler> = detail::dummy>
  Option *add_option(std::string option_name,
                     AssignTo &variable, ///< The variable to set
                     std::string option_description = "") {

    auto fun = [&variable](const CLI::results_t &res) { // comment for spacing
      return detail::lexical_conversion<AssignTo, ConvertTo>(res, variable);
    };

    Option *opt = add_option(option_name, fun, option_description, false, [&variable]() {
      return CLI::detail::checked_to_string<AssignTo, ConvertTo>(variable);
    });
    opt->type_name(detail::type_name<ConvertTo>());
    // these must be actual lvalues since (std::max) sometimes is defined in terms of references and references
    // to structs used in the evaluation can be temporary so that would cause issues.
    auto Tcount = detail::type_count<AssignTo>::value;
    auto XCcount = detail::type_count<ConvertTo>::value;
    opt->type_size(detail::type_count_min<ConvertTo>::value, (std::max)(Tcount, XCcount));
    opt->expected(detail::expected_count<ConvertTo>::value);
    opt->run_callback_for_default();
    return opt;
  }

  /// Add option for assigning to a variable
  template <typename AssignTo, enable_if_t<!std::is_const<AssignTo>::value, detail::enabler> = detail::dummy>
  Option *add_option_no_stream(std::string option_name,
                               AssignTo &variable, ///< The variable to set
                               std::string option_description = "") {

    auto fun = [&variable](const CLI::results_t &res) { // comment for spacing
      return detail::lexical_conversion<AssignTo, AssignTo>(res, variable);
    };

    Option *opt = add_option(option_name, fun, option_description, false, []() { return std::string{}; });
    opt->type_name(detail::type_name<AssignTo>());
    opt->type_size(detail::type_count_min<AssignTo>::value, detail::type_count<AssignTo>::value);
    opt->expected(detail::expected_count<AssignTo>::value);
    opt->run_callback_for_default();
    return opt;
  }

  /// Add option for a callback of a specific type
  template <typename ArgType>
  Option *add_option_function(std::string option_name,
                              const std::function<void(const ArgType &)> &func, ///< the callback to execute
                              std::string option_description = "") {

    auto fun = [func](const CLI::results_t &res) {
      ArgType variable;
      bool result = detail::lexical_conversion<ArgType, ArgType>(res, variable);
      if (result) {
        func(variable);
      }
      return result;
    };

    Option *opt = add_option(option_name, std::move(fun), option_description, false);
    opt->type_name(detail::type_name<ArgType>());
    opt->type_size(detail::type_count_min<ArgType>::value, detail::type_count<ArgType>::value);
    opt->expected(detail::expected_count<ArgType>::value);
    return opt;
  }

  /// Add option with no description or variable assignment
  Option *add_option(std::string option_name) {
    return add_option(option_name, CLI::callback_t{}, std::string{}, false);
  }

  /// Add option with description but with no variable assignment or callback
  template <typename T,
            enable_if_t<std::is_const<T>::value && std::is_constructible<std::string, T>::value, detail::enabler> =
                detail::dummy>
  Option *add_option(std::string option_name, T &option_description) {
    return add_option(option_name, CLI::callback_t(), option_description, false);
  }

  /// Set a help flag, replace the existing one if present
  Option *set_help_flag(std::string flag_name = "", const std::string &help_description = "");

  /// Set a help all flag, replaced the existing one if present
  Option *set_help_all_flag(std::string help_name = "", const std::string &help_description = "");

  /// Set a version flag and version display string, replace the existing one if present
  Option *set_version_flag(std::string flag_name = "",
                           const std::string &versionString = "",
                           const std::string &version_help = "Display program version information and exit");

  /// Generate the version string through a callback function
  Option *set_version_flag(std::string flag_name,
                           std::function<std::string()> vfunc,
                           const std::string &version_help = "Display program version information and exit");

private:
  /// Internal function for adding a flag
  Option *_add_flag_internal(std::string flag_name, CLI::callback_t fun, std::string flag_description);

public:
  /// Add a flag with no description or variable assignment
  Option *add_flag(std::string flag_name) { return _add_flag_internal(flag_name, CLI::callback_t(), std::string{}); }

  /// Add flag with description but with no variable assignment or callback
  /// takes a constant string,  if a variable string is passed that variable will be assigned the results from the
  /// flag
  template <typename T,
            enable_if_t<std::is_const<T>::value && std::is_constructible<std::string, T>::value, detail::enabler> =
                detail::dummy>
  Option *add_flag(std::string flag_name, T &flag_description) {
    return _add_flag_internal(flag_name, CLI::callback_t(), flag_description);
  }

  /// Other type version accepts all other types that are not vectors such as bool, enum, string or other classes
  /// that can be converted from a string
  template <typename T,
            enable_if_t<!detail::is_mutable_container<T>::value && !std::is_const<T>::value &&
                            !std::is_constructible<std::function<void(int)>, T>::value,
                        detail::enabler> = detail::dummy>
  Option *add_flag(std::string flag_name,
                   T &flag_result, ///< A variable holding the flag result
                   std::string flag_description = "") {

    CLI::callback_t fun = [&flag_result](const CLI::results_t &res) {
      using CLI::detail::lexical_cast;
      return lexical_cast(res[0], flag_result);
    };
    auto *opt = _add_flag_internal(flag_name, std::move(fun), std::move(flag_description));
    return detail::default_flag_modifiers<T>(opt);
  }

  /// Vector version to capture multiple flags.
  template <typename T,
            enable_if_t<!std::is_assignable<std::function<void(std::int64_t)> &, T>::value, detail::enabler> =
                detail::dummy>
  Option *add_flag(std::string flag_name,
                   std::vector<T> &flag_results, ///< A vector of values with the flag results
                   std::string flag_description = "") {
    CLI::callback_t fun = [&flag_results](const CLI::results_t &res) {
      bool retval = true;
      for (const auto &elem : res) {
        using CLI::detail::lexical_cast;
        flag_results.emplace_back();
        retval &= lexical_cast(elem, flag_results.back());
      }
      return retval;
    };
    return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
        ->multi_option_policy(MultiOptionPolicy::TakeAll)
        ->run_callback_for_default();
  }

  /// Add option for callback that is triggered with a true flag and takes no arguments
  Option *add_flag_callback(std::string flag_name,
                            std::function<void(void)> function, ///< A function to call, void(void)
                            std::string flag_description = "");

  /// Add option for callback with an integer value
  Option *add_flag_function(std::string flag_name,
                            std::function<void(std::int64_t)> function, ///< A function to call, void(int)
                            std::string flag_description = "");

#ifdef CLI11_CPP14
  /// Add option for callback (C++14 or better only)
  Option *add_flag(std::string flag_name,
                   std::function<void(std::int64_t)> function, ///< A function to call, void(std::int64_t)
                   std::string flag_description = "") {
    return add_flag_function(std::move(flag_name), std::move(function), std::move(flag_description));
  }
#endif

  /// Set a configuration ini file option, or clear it if no name passed
  Option *set_config(std::string option_name = "",
                     std::string default_filename = "",
                     const std::string &help_message = "Read an ini file",
                     bool config_required = false);

  /// Removes an option from the App. Takes an option pointer. Returns true if found and removed.
  bool remove_option(Option *opt);

  /// creates an option group as part of the given app
  template <typename T = Option_group>
  T *add_option_group(std::string group_name, std::string group_description = "") {
    if (!detail::valid_alias_name_string(group_name)) {
      throw IncorrectConstruction("option group names may not contain newlines or null characters");
    }
    auto option_group = std::make_shared<T>(std::move(group_description), group_name, this);
    auto *ptr = option_group.get();
    // move to App_p for overload resolution on older gcc versions
    App_p app_ptr = std::dynamic_pointer_cast<App>(option_group);
    add_subcommand(std::move(app_ptr));
    return ptr;
  }

  ///@}
  /// @name Subcommands
  ///@{

  /// Add a subcommand. Inherits INHERITABLE and OptionDefaults, and help flag
  App *add_subcommand(std::string subcommand_name = "", std::string subcommand_description = "");

  /// Add a previously created app as a subcommand
  App *add_subcommand(CLI::App_p subcom);

  /// Removes a subcommand from the App. Takes a subcommand pointer. Returns true if found and removed.
  bool remove_subcommand(App *subcom);

  /// Check to see if a subcommand is part of this command (doesn't have to be in command line)
  /// returns the first subcommand if passed a nullptr
  App *get_subcommand(const App *subcom) const;

  /// Check to see if a subcommand is part of this command (text version)
  CLI11_NODISCARD App *get_subcommand(std::string subcom) const;

  /// Get a subcommand by name (noexcept non-const version)
  /// returns null if subcommand doesn't exist
  CLI11_NODISCARD App *get_subcommand_no_throw(std::string subcom) const noexcept;

  /// Get a pointer to subcommand by index
  CLI11_NODISCARD App *get_subcommand(int index = 0) const;

  /// Check to see if a subcommand is part of this command and get a shared_ptr to it
  CLI::App_p get_subcommand_ptr(App *subcom) const;

  /// Check to see if a subcommand is part of this command (text version)
  CLI11_NODISCARD CLI::App_p get_subcommand_ptr(std::string subcom) const;

  /// Get an owning pointer to subcommand by index
  CLI11_NODISCARD CLI::App_p get_subcommand_ptr(int index = 0) const;

  /// Check to see if an option group is part of this App
  CLI11_NODISCARD App *get_option_group(std::string group_name) const;

  /// No argument version of count counts the number of times this subcommand was
  /// passed in. The main app will return 1. Unnamed subcommands will also return 1 unless
  /// otherwise modified in a callback
  CLI11_NODISCARD std::size_t count() const { return parsed_; }

  /// Get a count of all the arguments processed in options and subcommands, this excludes arguments which were
  /// treated as extras.
  CLI11_NODISCARD std::size_t count_all() const;

  /// Changes the group membership
  App *group(std::string group_name) {
    group_ = group_name;
    return this;
  }

  /// The argumentless form of require subcommand requires 1 or more subcommands
  App *require_subcommand() {
    require_subcommand_min_ = 1;
    require_subcommand_max_ = 0;
    return this;
  }

  /// Require a subcommand to be given (does not affect help call)
  /// The number required can be given. Negative values indicate maximum
  /// number allowed (0 for any number). Max number inheritable.
  App *require_subcommand(int value) {
    if (value < 0) {
      require_subcommand_min_ = 0;
      require_subcommand_max_ = static_cast<std::size_t>(-value);
    } else {
      require_subcommand_min_ = static_cast<std::size_t>(value);
      require_subcommand_max_ = static_cast<std::size_t>(value);
    }
    return this;
  }

  /// Explicitly control the number of subcommands required. Setting 0
  /// for the max means unlimited number allowed. Max number inheritable.
  App *require_subcommand(std::size_t min, std::size_t max) {
    require_subcommand_min_ = min;
    require_subcommand_max_ = max;
    return this;
  }

  /// The argumentless form of require option requires 1 or more options be used
  App *require_option() {
    require_option_min_ = 1;
    require_option_max_ = 0;
    return this;
  }

  /// Require an option to be given (does not affect help call)
  /// The number required can be given. Negative values indicate maximum
  /// number allowed (0 for any number).
  App *require_option(int value) {
    if (value < 0) {
      require_option_min_ = 0;
      require_option_max_ = static_cast<std::size_t>(-value);
    } else {
      require_option_min_ = static_cast<std::size_t>(value);
      require_option_max_ = static_cast<std::size_t>(value);
    }
    return this;
  }

  /// Explicitly control the number of options required. Setting 0
  /// for the max means unlimited number allowed. Max number inheritable.
  App *require_option(std::size_t min, std::size_t max) {
    require_option_min_ = min;
    require_option_max_ = max;
    return this;
  }

  /// Stop subcommand fallthrough, so that parent commands cannot collect commands after subcommand.
  /// Default from parent, usually set on parent.
  App *fallthrough(bool value = true) {
    fallthrough_ = value;
    return this;
  }

  /// Check to see if this subcommand was parsed, true only if received on command line.
  /// This allows the subcommand to be directly checked.
  explicit operator bool() const { return parsed_ > 0; }

  ///@}
  /// @name Extras for subclassing
  ///@{

  /// This allows subclasses to inject code before callbacks but after parse.
  ///
  /// This does not run if any errors or help is thrown.
  virtual void pre_callback() {}

  ///@}
  /// @name Parsing
  ///@{
  //
  /// Reset the parsed data
  void clear();

  /// Parses the command line - throws errors.
  /// This must be called after the options are in but before the rest of the program.
  void parse(int argc, const char *const *argv);
  void parse(int argc, const wchar_t *const *argv);

private:
  template <class CharT>
  void parse_char_t(int argc, const CharT *const *argv);

public:
  /// Parse a single string as if it contained command line arguments.
  /// This function splits the string into arguments then calls parse(std::vector<std::string> &)
  /// the function takes an optional boolean argument specifying if the programName is included in the string to
  /// process
  void parse(std::string commandline, bool program_name_included = false);
  void parse(std::wstring commandline, bool program_name_included = false);

  /// The real work is done here. Expects a reversed vector.
  /// Changes the vector to the remaining options.
  void parse(std::vector<std::string> &args);

  /// The real work is done here. Expects a reversed vector.
  void parse(std::vector<std::string> &&args);

  void parse_from_stream(std::istream &input);

  /// Provide a function to print a help message. The function gets access to the App pointer and error.
  void failure_message(std::function<std::string(const App *, const Error &e)> function) {
    failure_message_ = function;
  }

  /// Print a nice error message and return the exit code
  int exit(const Error &e, std::ostream &out = std::cout, std::ostream &err = std::cerr) const;

  ///@}
  /// @name Post parsing
  ///@{

  /// Counts the number of times the given option was passed.
  CLI11_NODISCARD std::size_t count(std::string option_name) const { return get_option(option_name)->count(); }

  /// Get a subcommand pointer list to the currently selected subcommands (after parsing by default, in command
  /// line order; use parsed = false to get the original definition list.)
  CLI11_NODISCARD std::vector<App *> get_subcommands() const { return parsed_subcommands_; }

  /// Get a filtered subcommand pointer list from the original definition list. An empty function will provide all
  /// subcommands (const)
  std::vector<const App *> get_subcommands(const std::function<bool(const App *)> &filter) const;

  /// Get a filtered subcommand pointer list from the original definition list. An empty function will provide all
  /// subcommands
  std::vector<App *> get_subcommands(const std::function<bool(App *)> &filter);

  /// Check to see if given subcommand was selected
  bool got_subcommand(const App *subcom) const {
    // get subcom needed to verify that this was a real subcommand
    return get_subcommand(subcom)->parsed_ > 0;
  }

  /// Check with name instead of pointer to see if subcommand was selected
  CLI11_NODISCARD bool got_subcommand(std::string subcommand_name) const noexcept {
    App *sub = get_subcommand_no_throw(subcommand_name);
    return (sub != nullptr) ? (sub->parsed_ > 0) : false;
  }

  /// Sets excluded options for the subcommand
  App *excludes(Option *opt) {
    if (opt == nullptr) {
      throw OptionNotFound("nullptr passed");
    }
    exclude_options_.insert(opt);
    return this;
  }

  /// Sets excluded subcommands for the subcommand
  App *excludes(App *app) {
    if (app == nullptr) {
      throw OptionNotFound("nullptr passed");
    }
    if (app == this) {
      throw OptionNotFound("cannot self reference in needs");
    }
    auto res = exclude_subcommands_.insert(app);
    // subcommand exclusion should be symmetric
    if (res.second) {
      app->exclude_subcommands_.insert(this);
    }
    return this;
  }

  App *needs(Option *opt) {
    if (opt == nullptr) {
      throw OptionNotFound("nullptr passed");
    }
    need_options_.insert(opt);
    return this;
  }

  App *needs(App *app) {
    if (app == nullptr) {
      throw OptionNotFound("nullptr passed");
    }
    if (app == this) {
      throw OptionNotFound("cannot self reference in needs");
    }
    need_subcommands_.insert(app);
    return this;
  }

  /// Removes an option from the excludes list of this subcommand
  bool remove_excludes(Option *opt);

  /// Removes a subcommand from the excludes list of this subcommand
  bool remove_excludes(App *app);

  /// Removes an option from the needs list of this subcommand
  bool remove_needs(Option *opt);

  /// Removes a subcommand from the needs list of this subcommand
  bool remove_needs(App *app);
  ///@}
  /// @name Help
  ///@{

  /// Set usage.
  App *usage(std::string usage_string) {
    usage_ = std::move(usage_string);
    return this;
  }
  /// Set usage.
  App *usage(std::function<std::string()> usage_function) {
    usage_callback_ = std::move(usage_function);
    return this;
  }
  /// Set footer.
  App *footer(std::string footer_string) {
    footer_ = std::move(footer_string);
    return this;
  }
  /// Set footer.
  App *footer(std::function<std::string()> footer_function) {
    footer_callback_ = std::move(footer_function);
    return this;
  }
  /// Produce a string that could be read in as a config of the current values of the App. Set default_also to
  /// include default arguments. write_descriptions will print a description for the App and for each option.
  CLI11_NODISCARD std::string config_to_str(bool default_also = false, bool write_description = false) const {
    return config_formatter_->to_config(this, default_also, write_description, "");
  }

  /// Makes a help message, using the currently configured formatter
  /// Will only do one subcommand at a time
  CLI11_NODISCARD std::string help(std::string prev = "", AppFormatMode mode = AppFormatMode::Normal) const;

  /// Displays a version string
  CLI11_NODISCARD std::string version() const;
  ///@}
  /// @name Getters
  ///@{

  /// Access the formatter
  CLI11_NODISCARD std::shared_ptr<FormatterBase> get_formatter() const { return formatter_; }

  /// Access the config formatter
  CLI11_NODISCARD std::shared_ptr<Config> get_config_formatter() const { return config_formatter_; }

  /// Access the config formatter as a configBase pointer
  CLI11_NODISCARD std::shared_ptr<ConfigBase> get_config_formatter_base() const {
    // This is safer as a dynamic_cast if we have RTTI, as Config -> ConfigBase
#if CLI11_USE_STATIC_RTTI == 0
    return std::dynamic_pointer_cast<ConfigBase>(config_formatter_);
#else
    return std::static_pointer_cast<ConfigBase>(config_formatter_);
#endif
  }

  /// Get the app or subcommand description
  CLI11_NODISCARD std::string get_description() const { return description_; }

  /// Set the description of the app
  App *description(std::string app_description) {
    description_ = std::move(app_description);
    return this;
  }

  /// Get the list of options (user facing function, so returns raw pointers), has optional filter function
  std::vector<const Option *> get_options(const std::function<bool(const Option *)> filter = {}) const;

  /// Non-const version of the above
  std::vector<Option *> get_options(const std::function<bool(Option *)> filter = {});

  /// Get an option by name (noexcept non-const version)
  CLI11_NODISCARD Option *get_option_no_throw(std::string option_name) noexcept;

  /// Get an option by name (noexcept const version)
  CLI11_NODISCARD const Option *get_option_no_throw(std::string option_name) const noexcept;

  /// Get an option by name
  CLI11_NODISCARD const Option *get_option(std::string option_name) const {
    const auto *opt = get_option_no_throw(option_name);
    if (opt == nullptr) {
      throw OptionNotFound(option_name);
    }
    return opt;
  }

  /// Get an option by name (non-const version)
  Option *get_option(std::string option_name) {
    auto *opt = get_option_no_throw(option_name);
    if (opt == nullptr) {
      throw OptionNotFound(option_name);
    }
    return opt;
  }

  /// Shortcut bracket operator for getting a pointer to an option
  const Option *operator[](const std::string &option_name) const { return get_option(option_name); }

  /// Shortcut bracket operator for getting a pointer to an option
  const Option *operator[](const char *option_name) const { return get_option(option_name); }

  /// Check the status of ignore_case
  CLI11_NODISCARD bool get_ignore_case() const { return ignore_case_; }

  /// Check the status of ignore_underscore
  CLI11_NODISCARD bool get_ignore_underscore() const { return ignore_underscore_; }

  /// Check the status of fallthrough
  CLI11_NODISCARD bool get_fallthrough() const { return fallthrough_; }

  /// Check the status of the allow windows style options
  CLI11_NODISCARD bool get_allow_windows_style_options() const { return allow_windows_style_options_; }

  /// Check the status of the allow windows style options
  CLI11_NODISCARD bool get_positionals_at_end() const { return positionals_at_end_; }

  /// Check the status of the allow windows style options
  CLI11_NODISCARD bool get_configurable() const { return configurable_; }

  /// Get the group of this subcommand
  CLI11_NODISCARD const std::string &get_group() const { return group_; }

  /// Generate and return the usage.
  CLI11_NODISCARD std::string get_usage() const {
    return (usage_callback_) ? usage_callback_() + '\n' + usage_ : usage_;
  }

  /// Generate and return the footer.
  CLI11_NODISCARD std::string get_footer() const {
    return (footer_callback_) ? footer_callback_() + '\n' + footer_ : footer_;
  }

  /// Get the required min subcommand value
  CLI11_NODISCARD std::size_t get_require_subcommand_min() const { return require_subcommand_min_; }

  /// Get the required max subcommand value
  CLI11_NODISCARD std::size_t get_require_subcommand_max() const { return require_subcommand_max_; }

  /// Get the required min option value
  CLI11_NODISCARD std::size_t get_require_option_min() const { return require_option_min_; }

  /// Get the required max option value
  CLI11_NODISCARD std::size_t get_require_option_max() const { return require_option_max_; }

  /// Get the prefix command status
  CLI11_NODISCARD bool get_prefix_command() const { return prefix_command_; }

  /// Get the status of allow extras
  CLI11_NODISCARD bool get_allow_extras() const { return allow_extras_; }

  /// Get the status of required
  CLI11_NODISCARD bool get_required() const { return required_; }

  /// Get the status of disabled
  CLI11_NODISCARD bool get_disabled() const { return disabled_; }

  /// Get the status of silence
  CLI11_NODISCARD bool get_silent() const { return silent_; }

  /// Get the status of disabled
  CLI11_NODISCARD bool get_immediate_callback() const { return immediate_callback_; }

  /// Get the status of disabled by default
  CLI11_NODISCARD bool get_disabled_by_default() const { return (default_startup == startup_mode::disabled); }

  /// Get the status of disabled by default
  CLI11_NODISCARD bool get_enabled_by_default() const { return (default_startup == startup_mode::enabled); }
  /// Get the status of validating positionals
  CLI11_NODISCARD bool get_validate_positionals() const { return validate_positionals_; }
  /// Get the status of validating optional vector arguments
  CLI11_NODISCARD bool get_validate_optional_arguments() const { return validate_optional_arguments_; }

  /// Get the status of allow extras
  CLI11_NODISCARD config_extras_mode get_allow_config_extras() const { return allow_config_extras_; }

  /// Get a pointer to the help flag.
  Option *get_help_ptr() { return help_ptr_; }

  /// Get a pointer to the help flag. (const)
  CLI11_NODISCARD const Option *get_help_ptr() const { return help_ptr_; }

  /// Get a pointer to the help all flag. (const)
  CLI11_NODISCARD const Option *get_help_all_ptr() const { return help_all_ptr_; }

  /// Get a pointer to the config option.
  Option *get_config_ptr() { return config_ptr_; }

  /// Get a pointer to the config option. (const)
  CLI11_NODISCARD const Option *get_config_ptr() const { return config_ptr_; }

  /// Get a pointer to the version option.
  Option *get_version_ptr() { return version_ptr_; }

  /// Get a pointer to the version option. (const)
  CLI11_NODISCARD const Option *get_version_ptr() const { return version_ptr_; }

  /// Get the parent of this subcommand (or nullptr if main app)
  App *get_parent() { return parent_; }

  /// Get the parent of this subcommand (or nullptr if main app) (const version)
  CLI11_NODISCARD const App *get_parent() const { return parent_; }

  /// Get the name of the current app
  CLI11_NODISCARD const std::string &get_name() const { return name_; }

  /// Get the aliases of the current app
  CLI11_NODISCARD const std::vector<std::string> &get_aliases() const { return aliases_; }

  /// clear all the aliases of the current App
  App *clear_aliases() {
    aliases_.clear();
    return this;
  }

  /// Get a display name for an app
  CLI11_NODISCARD std::string get_display_name(bool with_aliases = false) const;

  /// Check the name, case insensitive and underscore insensitive if set
  CLI11_NODISCARD bool check_name(std::string name_to_check) const;

  /// Get the groups available directly from this option (in order)
  CLI11_NODISCARD std::vector<std::string> get_groups() const;

  /// This gets a vector of pointers with the original parse order
  CLI11_NODISCARD const std::vector<Option *> &parse_order() const { return parse_order_; }

  /// This returns the missing options from the current subcommand
  CLI11_NODISCARD std::vector<std::string> remaining(bool recurse = false) const;

  /// This returns the missing options in a form ready for processing by another command line program
  CLI11_NODISCARD std::vector<std::string> remaining_for_passthrough(bool recurse = false) const;

  /// This returns the number of remaining options, minus the -- separator
  CLI11_NODISCARD std::size_t remaining_size(bool recurse = false) const;

  ///@}

protected:
  /// Check the options to make sure there are no conflicts.
  ///
  /// Currently checks to see if multiple positionals exist with unlimited args and checks if the min and max options
  /// are feasible
  void _validate() const;

  /// configure subcommands to enable parsing through the current object
  /// set the correct fallthrough and prefix for nameless subcommands and manage the automatic enable or disable
  /// makes sure parent is set correctly
  void _configure();

  /// Internal function to run (App) callback, bottom up
  void run_callback(bool final_mode = false, bool suppress_final_callback = false);

  /// Check to see if a subcommand is valid. Give up immediately if subcommand max has been reached.
  CLI11_NODISCARD bool _valid_subcommand(const std::string &current, bool ignore_used = true) const;

  /// Selects a Classifier enum based on the type of the current argument
  CLI11_NODISCARD detail::Classifier _recognize(const std::string &current,
                                                bool ignore_used_subcommands = true) const;

  // The parse function is now broken into several parts, and part of process

  /// Read and process a configuration file (main app only)
  void _process_config_file();

  /// Read and process a particular configuration file
  bool _process_config_file(const std::string &config_file, bool throw_error);

  /// Get envname options if not yet passed. Runs on *all* subcommands.
  void _process_env();

  /// Process callbacks. Runs on *all* subcommands.
  void _process_callbacks();

  /// Run help flag processing if any are found.
  ///
  /// The flags allow recursive calls to remember if there was a help flag on a parent.
  void _process_help_flags(bool trigger_help = false, bool trigger_all_help = false) const;

  /// Verify required options and cross requirements. Subcommands too (only if selected).
  void _process_requirements();

  /// Process callbacks and such.
  void _process();

  /// Throw an error if anything is left over and should not be.
  void _process_extras();

  /// Throw an error if anything is left over and should not be.
  /// Modifies the args to fill in the missing items before throwing.
  void _process_extras(std::vector<std::string> &args);

  /// Internal function to recursively increment the parsed counter on the current app as well unnamed subcommands
  void increment_parsed();

  /// Internal parse function
  void _parse(std::vector<std::string> &args);

  /// Internal parse function
  void _parse(std::vector<std::string> &&args);

  /// Internal function to parse a stream
  void _parse_stream(std::istream &input);

  /// Parse one config param, return false if not found in any subcommand, remove if it is
  ///
  /// If this has more than one dot.separated.name, go into the subcommand matching it
  /// Returns true if it managed to find the option, if false you'll need to remove the arg manually.
  void _parse_config(const std::vector<ConfigItem> &args);

  /// Fill in a single config option
  bool _parse_single_config(const ConfigItem &item, std::size_t level = 0);

  /// Parse "one" argument (some may eat more than one), delegate to parent if fails, add to missing if missing
  /// from main return false if the parse has failed and needs to return to parent
  bool _parse_single(std::vector<std::string> &args, bool &positional_only);

  /// Count the required remaining positional arguments
  CLI11_NODISCARD std::size_t _count_remaining_positionals(bool required_only = false) const;

  /// Count the required remaining positional arguments
  CLI11_NODISCARD bool _has_remaining_positionals() const;

  /// Parse a positional, go up the tree to check
  /// @param haltOnSubcommand if set to true the operation will not process subcommands merely return false
  /// Return true if the positional was used false otherwise
  bool _parse_positional(std::vector<std::string> &args, bool haltOnSubcommand);

  /// Locate a subcommand by name with two conditions, should disabled subcommands be ignored, and should used
  /// subcommands be ignored
  CLI11_NODISCARD App *
  _find_subcommand(const std::string &subc_name, bool ignore_disabled, bool ignore_used) const noexcept;

  /// Parse a subcommand, modify args and continue
  ///
  /// Unlike the others, this one will always allow fallthrough
  /// return true if the subcommand was processed false otherwise
  bool _parse_subcommand(std::vector<std::string> &args);

  /// Parse a short (false) or long (true) argument, must be at the top of the list
  /// if local_processing_only is set to true then fallthrough is disabled will return false if not found
  /// return true if the argument was processed or false if nothing was done
  bool _parse_arg(std::vector<std::string> &args, detail::Classifier current_type, bool local_processing_only);

  /// Trigger the pre_parse callback if needed
  void _trigger_pre_parse(std::size_t remaining_args);

  /// Get the appropriate parent to fallthrough to which is the first one that has a name or the main app
  App *_get_fallthrough_parent();

  /// Helper function to run through all possible comparisons of subcommand names to check there is no overlap
  CLI11_NODISCARD const std::string &_compare_subcommand_names(const App &subcom, const App &base) const;

  /// Helper function to place extra values in the most appropriate position
  void _move_to_missing(detail::Classifier val_type, const std::string &val);

public:
  /// function that could be used by subclasses of App to shift options around into subcommands
  void _move_option(Option *opt, App *app);
}; // namespace CLI

/// Extension of App to better manage groups of options
class Option_group : public App {
public:
  Option_group(std::string group_description, std::string group_name, App *parent)
      : App(std::move(group_description), "", parent) {
    group(group_name);
    // option groups should have automatic fallthrough
  }
  using App::add_option;
  /// Add an existing option to the Option_group
  Option *add_option(Option *opt) {
    if (get_parent() == nullptr) {
      throw OptionNotFound("Unable to locate the specified option");
    }
    get_parent()->_move_option(opt, this);
    return opt;
  }
  /// Add an existing option to the Option_group
  void add_options(Option *opt) { add_option(opt); }
  /// Add a bunch of options to the group
  template <typename... Args>
  void add_options(Option *opt, Args... args) {
    add_option(opt);
    add_options(args...);
  }
  using App::add_subcommand;
  /// Add an existing subcommand to be a member of an option_group
  App *add_subcommand(App *subcom) {
    App_p subc = subcom->get_parent()->get_subcommand_ptr(subcom);
    subc->get_parent()->remove_subcommand(subcom);
    add_subcommand(std::move(subc));
    return subcom;
  }
};

/// Helper function to enable one option group/subcommand when another is used
CLI11_INLINE void TriggerOn(App *trigger_app, App *app_to_enable);

/// Helper function to enable one option group/subcommand when another is used
CLI11_INLINE void TriggerOn(App *trigger_app, std::vector<App *> apps_to_enable);

/// Helper function to disable one option group/subcommand when another is used
CLI11_INLINE void TriggerOff(App *trigger_app, App *app_to_enable);

/// Helper function to disable one option group/subcommand when another is used
CLI11_INLINE void TriggerOff(App *trigger_app, std::vector<App *> apps_to_enable);

/// Helper function to mark an option as deprecated
CLI11_INLINE void deprecate_option(Option *opt, const std::string &replacement = "");

/// Helper function to mark an option as deprecated
inline void deprecate_option(App *app, const std::string &option_name, const std::string &replacement = "") {
  auto *opt = app->get_option(option_name);
  deprecate_option(opt, replacement);
}

/// Helper function to mark an option as deprecated
inline void deprecate_option(App &app, const std::string &option_name, const std::string &replacement = "") {
  auto *opt = app.get_option(option_name);
  deprecate_option(opt, replacement);
}

/// Helper function to mark an option as retired
CLI11_INLINE void retire_option(App *app, Option *opt);

/// Helper function to mark an option as retired
CLI11_INLINE void retire_option(App &app, Option *opt);

/// Helper function to mark an option as retired
CLI11_INLINE void retire_option(App *app, const std::string &option_name);

/// Helper function to mark an option as retired
CLI11_INLINE void retire_option(App &app, const std::string &option_name);

namespace detail {
/// This class is simply to allow tests access to App's protected functions
struct AppFriend {
#ifdef CLI11_CPP14

  /// Wrap _parse_short, perfectly forward arguments and return
  template <typename... Args>
  static decltype(auto) parse_arg(App *app, Args &&...args) {
    return app->_parse_arg(std::forward<Args>(args)...);
  }

  /// Wrap _parse_subcommand, perfectly forward arguments and return
  template <typename... Args>
  static decltype(auto) parse_subcommand(App *app, Args &&...args) {
    return app->_parse_subcommand(std::forward<Args>(args)...);
  }
#else
  /// Wrap _parse_short, perfectly forward arguments and return
  template <typename... Args>
  static auto parse_arg(App *app, Args &&...args) ->
      typename std::result_of<decltype (&App::_parse_arg)(App, Args...)>::type {
    return app->_parse_arg(std::forward<Args>(args)...);
  }

  /// Wrap _parse_subcommand, perfectly forward arguments and return
  template <typename... Args>
  static auto parse_subcommand(App *app, Args &&...args) ->
      typename std::result_of<decltype (&App::_parse_subcommand)(App, Args...)>::type {
    return app->_parse_subcommand(std::forward<Args>(args)...);
  }
#endif
  /// Wrap the fallthrough parent function to make sure that is working correctly
  static App *get_fallthrough_parent(App *app) { return app->_get_fallthrough_parent(); }
};
} // namespace detail

CLI11_INLINE App::App(std::string app_description, std::string app_name, App *parent)
    : name_(std::move(app_name)), description_(std::move(app_description)), parent_(parent) {
  // Inherit if not from a nullptr
  if (parent_ != nullptr) {
    if (parent_->help_ptr_ != nullptr)
      set_help_flag(parent_->help_ptr_->get_name(false, true), parent_->help_ptr_->get_description());
    if (parent_->help_all_ptr_ != nullptr)
      set_help_all_flag(parent_->help_all_ptr_->get_name(false, true), parent_->help_all_ptr_->get_description());

    /// OptionDefaults
    option_defaults_ = parent_->option_defaults_;

    // INHERITABLE
    failure_message_ = parent_->failure_message_;
    allow_extras_ = parent_->allow_extras_;
    allow_config_extras_ = parent_->allow_config_extras_;
    prefix_command_ = parent_->prefix_command_;
    immediate_callback_ = parent_->immediate_callback_;
    ignore_case_ = parent_->ignore_case_;
    ignore_underscore_ = parent_->ignore_underscore_;
    fallthrough_ = parent_->fallthrough_;
    validate_positionals_ = parent_->validate_positionals_;
    validate_optional_arguments_ = parent_->validate_optional_arguments_;
    configurable_ = parent_->configurable_;
    allow_windows_style_options_ = parent_->allow_windows_style_options_;
    group_ = parent_->group_;
    usage_ = parent_->usage_;
    footer_ = parent_->footer_;
    formatter_ = parent_->formatter_;
    config_formatter_ = parent_->config_formatter_;
    require_subcommand_max_ = parent_->require_subcommand_max_;
  }
}

CLI11_NODISCARD CLI11_INLINE char **App::ensure_utf8(char **argv) {
#ifdef _WIN32
  (void)argv;

  normalized_argv_ = detail::compute_win32_argv();

  if (!normalized_argv_view_.empty()) {
    normalized_argv_view_.clear();
  }

  normalized_argv_view_.reserve(normalized_argv_.size());
  for (auto &arg : normalized_argv_) {
    // using const_cast is well-defined, string is known to not be const.
    normalized_argv_view_.push_back(const_cast<char *>(arg.data()));
  }

  return normalized_argv_view_.data();
#else
  return argv;
#endif
}

CLI11_INLINE App *App::name(std::string app_name) {

  if (parent_ != nullptr) {
    std::string oname = name_;
    name_ = app_name;
    const auto &res = _compare_subcommand_names(*this, *_get_fallthrough_parent());
    if (!res.empty()) {
      name_ = oname;
      throw(OptionAlreadyAdded(app_name + " conflicts with existing subcommand names"));
    }
  } else {
    name_ = app_name;
  }
  has_automatic_name_ = false;
  return this;
}

CLI11_INLINE App *App::alias(std::string app_name) {
  if (app_name.empty() || !detail::valid_alias_name_string(app_name)) {
    throw IncorrectConstruction("Aliases may not be empty or contain newlines or null characters");
  }
  if (parent_ != nullptr) {
    aliases_.push_back(app_name);
    const auto &res = _compare_subcommand_names(*this, *_get_fallthrough_parent());
    if (!res.empty()) {
      aliases_.pop_back();
      throw(OptionAlreadyAdded("alias already matches an existing subcommand: " + app_name));
    }
  } else {
    aliases_.push_back(app_name);
  }

  return this;
}

CLI11_INLINE App *App::immediate_callback(bool immediate) {
  immediate_callback_ = immediate;
  if (immediate_callback_) {
    if (final_callback_ && !(parse_complete_callback_)) {
      std::swap(final_callback_, parse_complete_callback_);
    }
  } else if (!(final_callback_) && parse_complete_callback_) {
    std::swap(final_callback_, parse_complete_callback_);
  }
  return this;
}

CLI11_INLINE App *App::ignore_case(bool value) {
  if (value && !ignore_case_) {
    ignore_case_ = true;
    auto *p = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
    const auto &match = _compare_subcommand_names(*this, *p);
    if (!match.empty()) {
      ignore_case_ = false; // we are throwing so need to be exception invariant
      throw OptionAlreadyAdded("ignore case would cause subcommand name conflicts: " + match);
    }
  }
  ignore_case_ = value;
  return this;
}

CLI11_INLINE App *App::ignore_underscore(bool value) {
  if (value && !ignore_underscore_) {
    ignore_underscore_ = true;
    auto *p = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
    const auto &match = _compare_subcommand_names(*this, *p);
    if (!match.empty()) {
      ignore_underscore_ = false;
      throw OptionAlreadyAdded("ignore underscore would cause subcommand name conflicts: " + match);
    }
  }
  ignore_underscore_ = value;
  return this;
}

CLI11_INLINE Option *App::add_option(std::string option_name,
                                     callback_t option_callback,
                                     std::string option_description,
                                     bool defaulted,
                                     std::function<std::string()> func) {
  Option myopt{option_name, option_description, option_callback, this};

  if (std::find_if(std::begin(options_), std::end(options_), [&myopt](const Option_p &v) { return *v == myopt; }) ==
      std::end(options_)) {
    if (myopt.lnames_.empty() && myopt.snames_.empty()) {
      // if the option is positional only there is additional potential for ambiguities in config files and needs
      // to be checked
      std::string test_name = "--" + myopt.get_single_name();
      if (test_name.size() == 3) {
        test_name.erase(0, 1);
      }

      auto *op = get_option_no_throw(test_name);
      if (op != nullptr) {
        throw(OptionAlreadyAdded("added option positional name matches existing option: " + test_name));
      }
    } else if (parent_ != nullptr) {
      for (auto &ln : myopt.lnames_) {
        auto *op = parent_->get_option_no_throw(ln);
        if (op != nullptr) {
          throw(OptionAlreadyAdded("added option matches existing positional option: " + ln));
        }
      }
      for (auto &sn : myopt.snames_) {
        auto *op = parent_->get_option_no_throw(sn);
        if (op != nullptr) {
          throw(OptionAlreadyAdded("added option matches existing positional option: " + sn));
        }
      }
    }
    options_.emplace_back();
    Option_p &option = options_.back();
    option.reset(new Option(option_name, option_description, option_callback, this));

    // Set the default string capture function
    option->default_function(func);

    // For compatibility with CLI11 1.7 and before, capture the default string here
    if (defaulted)
      option->capture_default_str();

    // Transfer defaults to the new option
    option_defaults_.copy_to(option.get());

    // Don't bother to capture if we already did
    if (!defaulted && option->get_always_capture_default())
      option->capture_default_str();

    return option.get();
  }
  // we know something matches now find what it is so we can produce more error information
  for (auto &opt : options_) {
    const auto &matchname = opt->matching_name(myopt);
    if (!matchname.empty()) {
      throw(OptionAlreadyAdded("added option matched existing option name: " + matchname));
    }
  }
  // this line should not be reached the above loop should trigger the throw
  throw(OptionAlreadyAdded("added option matched existing option name")); // LCOV_EXCL_LINE
}

CLI11_INLINE Option *App::set_help_flag(std::string flag_name, const std::string &help_description) {
  // take flag_description by const reference otherwise add_flag tries to assign to help_description
  if (help_ptr_ != nullptr) {
    remove_option(help_ptr_);
    help_ptr_ = nullptr;
  }

  // Empty name will simply remove the help flag
  if (!flag_name.empty()) {
    help_ptr_ = add_flag(flag_name, help_description);
    help_ptr_->configurable(false);
  }

  return help_ptr_;
}

CLI11_INLINE Option *App::set_help_all_flag(std::string help_name, const std::string &help_description) {
  // take flag_description by const reference otherwise add_flag tries to assign to flag_description
  if (help_all_ptr_ != nullptr) {
    remove_option(help_all_ptr_);
    help_all_ptr_ = nullptr;
  }

  // Empty name will simply remove the help all flag
  if (!help_name.empty()) {
    help_all_ptr_ = add_flag(help_name, help_description);
    help_all_ptr_->configurable(false);
  }

  return help_all_ptr_;
}

CLI11_INLINE Option *
App::set_version_flag(std::string flag_name, const std::string &versionString, const std::string &version_help) {
  // take flag_description by const reference otherwise add_flag tries to assign to version_description
  if (version_ptr_ != nullptr) {
    remove_option(version_ptr_);
    version_ptr_ = nullptr;
  }

  // Empty name will simply remove the version flag
  if (!flag_name.empty()) {
    version_ptr_ = add_flag_callback(
        flag_name, [versionString]() { throw(CLI::CallForVersion(versionString, 0)); }, version_help);
    version_ptr_->configurable(false);
  }

  return version_ptr_;
}

CLI11_INLINE Option *
App::set_version_flag(std::string flag_name, std::function<std::string()> vfunc, const std::string &version_help) {
  if (version_ptr_ != nullptr) {
    remove_option(version_ptr_);
    version_ptr_ = nullptr;
  }

  // Empty name will simply remove the version flag
  if (!flag_name.empty()) {
    version_ptr_ =
        add_flag_callback(flag_name, [vfunc]() { throw(CLI::CallForVersion(vfunc(), 0)); }, version_help);
    version_ptr_->configurable(false);
  }

  return version_ptr_;
}

CLI11_INLINE Option *App::_add_flag_internal(std::string flag_name, CLI::callback_t fun, std::string flag_description) {
  Option *opt = nullptr;
  if (detail::has_default_flag_values(flag_name)) {
    // check for default values and if it has them
    auto flag_defaults = detail::get_default_flag_values(flag_name);
    detail::remove_default_flag_values(flag_name);
    opt = add_option(std::move(flag_name), std::move(fun), std::move(flag_description), false);
    for (const auto &fname : flag_defaults)
      opt->fnames_.push_back(fname.first);
    opt->default_flag_values_ = std::move(flag_defaults);
  } else {
    opt = add_option(std::move(flag_name), std::move(fun), std::move(flag_description), false);
  }
  // flags cannot have positional values
  if (opt->get_positional()) {
    auto pos_name = opt->get_name(true);
    remove_option(opt);
    throw IncorrectConstruction::PositionalFlag(pos_name);
  }
  opt->multi_option_policy(MultiOptionPolicy::TakeLast);
  opt->expected(0);
  opt->required(false);
  return opt;
}

CLI11_INLINE Option *App::add_flag_callback(std::string flag_name,
                                            std::function<void(void)> function, ///< A function to call, void(void)
                                            std::string flag_description) {

  CLI::callback_t fun = [function](const CLI::results_t &res) {
    using CLI::detail::lexical_cast;
    bool trigger{false};
    auto result = lexical_cast(res[0], trigger);
    if (result && trigger) {
      function();
    }
    return result;
  };
  return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description));
}

CLI11_INLINE Option *
App::add_flag_function(std::string flag_name,
                       std::function<void(std::int64_t)> function, ///< A function to call, void(int)
                       std::string flag_description) {

  CLI::callback_t fun = [function](const CLI::results_t &res) {
    using CLI::detail::lexical_cast;
    std::int64_t flag_count{0};
    lexical_cast(res[0], flag_count);
    function(flag_count);
    return true;
  };
  return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
      ->multi_option_policy(MultiOptionPolicy::Sum);
}

CLI11_INLINE Option *App::set_config(std::string option_name,
                                     std::string default_filename,
                                     const std::string &help_message,
                                     bool config_required) {

  // Remove existing config if present
  if (config_ptr_ != nullptr) {
    remove_option(config_ptr_);
    config_ptr_ = nullptr; // need to remove the config_ptr completely
  }

  // Only add config if option passed
  if (!option_name.empty()) {
    config_ptr_ = add_option(option_name, help_message);
    if (config_required) {
      config_ptr_->required();
    }
    if (!default_filename.empty()) {
      config_ptr_->default_str(std::move(default_filename));
      config_ptr_->force_callback_ = true;
    }
    config_ptr_->configurable(false);
    // set the option to take the last value and reverse given by default
    config_ptr_->multi_option_policy(MultiOptionPolicy::Reverse);
  }

  return config_ptr_;
}

CLI11_INLINE bool App::remove_option(Option *opt) {
  // Make sure no links exist
  for (Option_p &op : options_) {
    op->remove_needs(opt);
    op->remove_excludes(opt);
  }

  if (help_ptr_ == opt)
    help_ptr_ = nullptr;
  if (help_all_ptr_ == opt)
    help_all_ptr_ = nullptr;

  auto iterator =
      std::find_if(std::begin(options_), std::end(options_), [opt](const Option_p &v) { return v.get() == opt; });
  if (iterator != std::end(options_)) {
    options_.erase(iterator);
    return true;
  }
  return false;
}

CLI11_INLINE App *App::add_subcommand(std::string subcommand_name, std::string subcommand_description) {
  if (!subcommand_name.empty() && !detail::valid_name_string(subcommand_name)) {
    if (!detail::valid_first_char(subcommand_name[0])) {
      throw IncorrectConstruction(
          "Subcommand name starts with invalid character, '!' and '-' and control characters");
    }
    for (auto c : subcommand_name) {
      if (!detail::valid_later_char(c)) {
        throw IncorrectConstruction(std::string("Subcommand name contains invalid character ('") + c +
                                    "'), all characters are allowed except"
                                    "'=',':','{','}', ' ', and control characters");
      }
    }
  }
  CLI::App_p subcom = std::shared_ptr<App>(new App(std::move(subcommand_description), subcommand_name, this));
  return add_subcommand(std::move(subcom));
}

CLI11_INLINE App *App::add_subcommand(CLI::App_p subcom) {
  if (!subcom)
    throw IncorrectConstruction("passed App is not valid");
  auto *ckapp = (name_.empty() && parent_ != nullptr) ? _get_fallthrough_parent() : this;
  const auto &mstrg = _compare_subcommand_names(*subcom, *ckapp);
  if (!mstrg.empty()) {
    throw(OptionAlreadyAdded("subcommand name or alias matches existing subcommand: " + mstrg));
  }
  subcom->parent_ = this;
  subcommands_.push_back(std::move(subcom));
  return subcommands_.back().get();
}

CLI11_INLINE bool App::remove_subcommand(App *subcom) {
  // Make sure no links exist
  for (App_p &sub : subcommands_) {
    sub->remove_excludes(subcom);
    sub->remove_needs(subcom);
  }

  auto iterator = std::find_if(
      std::begin(subcommands_), std::end(subcommands_), [subcom](const App_p &v) { return v.get() == subcom; });
  if (iterator != std::end(subcommands_)) {
    subcommands_.erase(iterator);
    return true;
  }
  return false;
}

CLI11_INLINE App *App::get_subcommand(const App *subcom) const {
  if (subcom == nullptr)
    throw OptionNotFound("nullptr passed");
  for (const App_p &subcomptr : subcommands_)
    if (subcomptr.get() == subcom)
      return subcomptr.get();
  throw OptionNotFound(subcom->get_name());
}

CLI11_NODISCARD CLI11_INLINE App *App::get_subcommand(std::string subcom) const {
  auto *subc = _find_subcommand(subcom, false, false);
  if (subc == nullptr)
    throw OptionNotFound(subcom);
  return subc;
}

CLI11_NODISCARD CLI11_INLINE App *App::get_subcommand_no_throw(std::string subcom) const noexcept {
  return _find_subcommand(subcom, false, false);
}

CLI11_NODISCARD CLI11_INLINE App *App::get_subcommand(int index) const {
  if (index >= 0) {
    auto uindex = static_cast<unsigned>(index);
    if (uindex < subcommands_.size())
      return subcommands_[uindex].get();
  }
  throw OptionNotFound(std::to_string(index));
}

CLI11_INLINE CLI::App_p App::get_subcommand_ptr(App *subcom) const {
  if (subcom == nullptr)
    throw OptionNotFound("nullptr passed");
  for (const App_p &subcomptr : subcommands_)
    if (subcomptr.get() == subcom)
      return subcomptr;
  throw OptionNotFound(subcom->get_name());
}

CLI11_NODISCARD CLI11_INLINE CLI::App_p App::get_subcommand_ptr(std::string subcom) const {
  for (const App_p &subcomptr : subcommands_)
    if (subcomptr->check_name(subcom))
      return subcomptr;
  throw OptionNotFound(subcom);
}

CLI11_NODISCARD CLI11_INLINE CLI::App_p App::get_subcommand_ptr(int index) const {
  if (index >= 0) {
    auto uindex = static_cast<unsigned>(index);
    if (uindex < subcommands_.size())
      return subcommands_[uindex];
  }
  throw OptionNotFound(std::to_string(index));
}

CLI11_NODISCARD CLI11_INLINE CLI::App *App::get_option_group(std::string group_name) const {
  for (const App_p &app : subcommands_) {
    if (app->name_.empty() && app->group_ == group_name) {
      return app.get();
    }
  }
  throw OptionNotFound(group_name);
}

CLI11_NODISCARD CLI11_INLINE std::size_t App::count_all() const {
  std::size_t cnt{0};
  for (const auto &opt : options_) {
    cnt += opt->count();
  }
  for (const auto &sub : subcommands_) {
    cnt += sub->count_all();
  }
  if (!get_name().empty()) { // for named subcommands add the number of times the subcommand was called
    cnt += parsed_;
  }
  return cnt;
}

CLI11_INLINE void App::clear() {

  parsed_ = 0;
  pre_parse_called_ = false;

  missing_.clear();
  parsed_subcommands_.clear();
  for (const Option_p &opt : options_) {
    opt->clear();
  }
  for (const App_p &subc : subcommands_) {
    subc->clear();
  }
}

CLI11_INLINE void App::parse(int argc, const char *const *argv) { parse_char_t(argc, argv); }
CLI11_INLINE void App::parse(int argc, const wchar_t *const *argv) { parse_char_t(argc, argv); }

namespace detail {

// Do nothing or perform narrowing
CLI11_INLINE const char *maybe_narrow(const char *str) { return str; }
CLI11_INLINE std::string maybe_narrow(const wchar_t *str) { return narrow(str); }

} // namespace detail

template <class CharT>
CLI11_INLINE void App::parse_char_t(int argc, const CharT *const *argv) {
  // If the name is not set, read from command line
  if (name_.empty() || has_automatic_name_) {
    has_automatic_name_ = true;
    name_ = detail::maybe_narrow(argv[0]);
  }

  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc) - 1U);
  for (auto i = static_cast<std::size_t>(argc) - 1U; i > 0U; --i)
    args.emplace_back(detail::maybe_narrow(argv[i]));

  parse(std::move(args));
}

CLI11_INLINE void App::parse(std::string commandline, bool program_name_included) {

  if (program_name_included) {
    auto nstr = detail::split_program_name(commandline);
    if ((name_.empty()) || (has_automatic_name_)) {
      has_automatic_name_ = true;
      name_ = nstr.first;
    }
    commandline = std::move(nstr.second);
  } else {
    detail::trim(commandline);
  }
  // the next section of code is to deal with quoted arguments after an '=' or ':' for windows like operations
  if (!commandline.empty()) {
    commandline = detail::find_and_modify(commandline, "=", detail::escape_detect);
    if (allow_windows_style_options_)
      commandline = detail::find_and_modify(commandline, ":", detail::escape_detect);
  }

  auto args = detail::split_up(std::move(commandline));
  // remove all empty strings
  args.erase(std::remove(args.begin(), args.end(), std::string{}), args.end());
  try {
    detail::remove_quotes(args);
  } catch (const std::invalid_argument &arg) {
    throw CLI::ParseError(arg.what(), CLI::ExitCodes::InvalidError);
  }
  std::reverse(args.begin(), args.end());
  parse(std::move(args));
}

CLI11_INLINE void App::parse(std::wstring commandline, bool program_name_included) {
  parse(narrow(commandline), program_name_included);
}

CLI11_INLINE void App::parse(std::vector<std::string> &args) {
  // Clear if parsed
  if (parsed_ > 0)
    clear();

  // parsed_ is incremented in commands/subcommands,
  // but placed here to make sure this is cleared when
  // running parse after an error is thrown, even by _validate or _configure.
  parsed_ = 1;
  _validate();
  _configure();
  // set the parent as nullptr as this object should be the top now
  parent_ = nullptr;
  parsed_ = 0;

  _parse(args);
  run_callback();
}

CLI11_INLINE void App::parse(std::vector<std::string> &&args) {
  // Clear if parsed
  if (parsed_ > 0)
    clear();

  // parsed_ is incremented in commands/subcommands,
  // but placed here to make sure this is cleared when
  // running parse after an error is thrown, even by _validate or _configure.
  parsed_ = 1;
  _validate();
  _configure();
  // set the parent as nullptr as this object should be the top now
  parent_ = nullptr;
  parsed_ = 0;

  _parse(std::move(args));
  run_callback();
}

CLI11_INLINE void App::parse_from_stream(std::istream &input) {
  if (parsed_ == 0) {
    _validate();
    _configure();
    // set the parent as nullptr as this object should be the top now
  }

  _parse_stream(input);
  run_callback();
}

CLI11_INLINE int App::exit(const Error &e, std::ostream &out, std::ostream &err) const {

  /// Avoid printing anything if this is a CLI::RuntimeError
  if (e.get_name() == "RuntimeError")
    return e.get_exit_code();

  if (e.get_name() == "CallForHelp") {
    out << help();
    return e.get_exit_code();
  }

  if (e.get_name() == "CallForAllHelp") {
    out << help("", AppFormatMode::All);
    return e.get_exit_code();
  }

  if (e.get_name() == "CallForVersion") {
    out << e.what() << '\n';
    return e.get_exit_code();
  }

  if (e.get_exit_code() != static_cast<int>(ExitCodes::Success)) {
    if (failure_message_)
      err << failure_message_(this, e) << std::flush;
  }

  return e.get_exit_code();
}

CLI11_INLINE std::vector<const App *> App::get_subcommands(const std::function<bool(const App *)> &filter) const {
  std::vector<const App *> subcomms(subcommands_.size());
  std::transform(
      std::begin(subcommands_), std::end(subcommands_), std::begin(subcomms), [](const App_p &v) { return v.get(); });

  if (filter) {
    subcomms.erase(std::remove_if(std::begin(subcomms),
                                  std::end(subcomms),
                                  [&filter](const App *app) { return !filter(app); }),
                   std::end(subcomms));
  }

  return subcomms;
}

CLI11_INLINE std::vector<App *> App::get_subcommands(const std::function<bool(App *)> &filter) {
  std::vector<App *> subcomms(subcommands_.size());
  std::transform(
      std::begin(subcommands_), std::end(subcommands_), std::begin(subcomms), [](const App_p &v) { return v.get(); });

  if (filter) {
    subcomms.erase(
        std::remove_if(std::begin(subcomms), std::end(subcomms), [&filter](App *app) { return !filter(app); }),
        std::end(subcomms));
  }

  return subcomms;
}

CLI11_INLINE bool App::remove_excludes(Option *opt) {
  auto iterator = std::find(std::begin(exclude_options_), std::end(exclude_options_), opt);
  if (iterator == std::end(exclude_options_)) {
    return false;
  }
  exclude_options_.erase(iterator);
  return true;
}

CLI11_INLINE bool App::remove_excludes(App *app) {
  auto iterator = std::find(std::begin(exclude_subcommands_), std::end(exclude_subcommands_), app);
  if (iterator == std::end(exclude_subcommands_)) {
    return false;
  }
  auto *other_app = *iterator;
  exclude_subcommands_.erase(iterator);
  other_app->remove_excludes(this);
  return true;
}

CLI11_INLINE bool App::remove_needs(Option *opt) {
  auto iterator = std::find(std::begin(need_options_), std::end(need_options_), opt);
  if (iterator == std::end(need_options_)) {
    return false;
  }
  need_options_.erase(iterator);
  return true;
}

CLI11_INLINE bool App::remove_needs(App *app) {
  auto iterator = std::find(std::begin(need_subcommands_), std::end(need_subcommands_), app);
  if (iterator == std::end(need_subcommands_)) {
    return false;
  }
  need_subcommands_.erase(iterator);
  return true;
}

CLI11_NODISCARD CLI11_INLINE std::string App::help(std::string prev, AppFormatMode mode) const {
  if (prev.empty())
    prev = get_name();
  else
    prev += " " + get_name();

  // Delegate to subcommand if needed
  auto selected_subcommands = get_subcommands();
  if (!selected_subcommands.empty()) {
    return selected_subcommands.back()->help(prev, mode);
  }
  return formatter_->make_help(this, prev, mode);
}

CLI11_NODISCARD CLI11_INLINE std::string App::version() const {
  std::string val;
  if (version_ptr_ != nullptr) {
    // copy the results for reuse later
    results_t rv = version_ptr_->results();
    version_ptr_->clear();
    version_ptr_->add_result("true");
    try {
      version_ptr_->run_callback();
    } catch (const CLI::CallForVersion &cfv) {
      val = cfv.what();
    }
    version_ptr_->clear();
    version_ptr_->add_result(rv);
  }
  return val;
}

CLI11_INLINE std::vector<const Option *> App::get_options(const std::function<bool(const Option *)> filter) const {
  std::vector<const Option *> options(options_.size());
  std::transform(
      std::begin(options_), std::end(options_), std::begin(options), [](const Option_p &val) { return val.get(); });

  if (filter) {
    options.erase(std::remove_if(std::begin(options),
                                 std::end(options),
                                 [&filter](const Option *opt) { return !filter(opt); }),
                  std::end(options));
  }

  return options;
}

CLI11_INLINE std::vector<Option *> App::get_options(const std::function<bool(Option *)> filter) {
  std::vector<Option *> options(options_.size());
  std::transform(
      std::begin(options_), std::end(options_), std::begin(options), [](const Option_p &val) { return val.get(); });

  if (filter) {
    options.erase(
        std::remove_if(std::begin(options), std::end(options), [&filter](Option *opt) { return !filter(opt); }),
        std::end(options));
  }

  return options;
}

CLI11_NODISCARD CLI11_INLINE Option *App::get_option_no_throw(std::string option_name) noexcept {
  for (Option_p &opt : options_) {
    if (opt->check_name(option_name)) {
      return opt.get();
    }
  }
  for (auto &subc : subcommands_) {
    // also check down into nameless subcommands
    if (subc->get_name().empty()) {
      auto *opt = subc->get_option_no_throw(option_name);
      if (opt != nullptr) {
        return opt;
      }
    }
  }
  return nullptr;
}

CLI11_NODISCARD CLI11_INLINE const Option *App::get_option_no_throw(std::string option_name) const noexcept {
  for (const Option_p &opt : options_) {
    if (opt->check_name(option_name)) {
      return opt.get();
    }
  }
  for (const auto &subc : subcommands_) {
    // also check down into nameless subcommands
    if (subc->get_name().empty()) {
      auto *opt = subc->get_option_no_throw(option_name);
      if (opt != nullptr) {
        return opt;
      }
    }
  }
  return nullptr;
}

CLI11_NODISCARD CLI11_INLINE std::string App::get_display_name(bool with_aliases) const {
  if (name_.empty()) {
    return std::string("[Option Group: ") + get_group() + "]";
  }
  if (aliases_.empty() || !with_aliases) {
    return name_;
  }
  std::string dispname = name_;
  for (const auto &lalias : aliases_) {
    dispname.push_back(',');
    dispname.push_back(' ');
    dispname.append(lalias);
  }
  return dispname;
}

CLI11_NODISCARD CLI11_INLINE bool App::check_name(std::string name_to_check) const {
  std::string local_name = name_;
  if (ignore_underscore_) {
    local_name = detail::remove_underscore(name_);
    name_to_check = detail::remove_underscore(name_to_check);
  }
  if (ignore_case_) {
    local_name = detail::to_lower(name_);
    name_to_check = detail::to_lower(name_to_check);
  }

  if (local_name == name_to_check) {
    return true;
  }
  for (std::string les : aliases_) { // NOLINT(performance-for-range-copy)
    if (ignore_underscore_) {
      les = detail::remove_underscore(les);
    }
    if (ignore_case_) {
      les = detail::to_lower(les);
    }
    if (les == name_to_check) {
      return true;
    }
  }
  return false;
}

CLI11_NODISCARD CLI11_INLINE std::vector<std::string> App::get_groups() const {
  std::vector<std::string> groups;

  for (const Option_p &opt : options_) {
    // Add group if it is not already in there
    if (std::find(groups.begin(), groups.end(), opt->get_group()) == groups.end()) {
      groups.push_back(opt->get_group());
    }
  }

  return groups;
}

CLI11_NODISCARD CLI11_INLINE std::vector<std::string> App::remaining(bool recurse) const {
  std::vector<std::string> miss_list;
  for (const std::pair<detail::Classifier, std::string> &miss : missing_) {
    miss_list.push_back(std::get<1>(miss));
  }
  // Get from a subcommand that may allow extras
  if (recurse) {
    if (!allow_extras_) {
      for (const auto &sub : subcommands_) {
        if (sub->name_.empty() && !sub->missing_.empty()) {
          for (const std::pair<detail::Classifier, std::string> &miss : sub->missing_) {
            miss_list.push_back(std::get<1>(miss));
          }
        }
      }
    }
    // Recurse into subcommands

    for (const App *sub : parsed_subcommands_) {
      std::vector<std::string> output = sub->remaining(recurse);
      std::copy(std::begin(output), std::end(output), std::back_inserter(miss_list));
    }
  }
  return miss_list;
}

CLI11_NODISCARD CLI11_INLINE std::vector<std::string> App::remaining_for_passthrough(bool recurse) const {
  std::vector<std::string> miss_list = remaining(recurse);
  std::reverse(std::begin(miss_list), std::end(miss_list));
  return miss_list;
}

CLI11_NODISCARD CLI11_INLINE std::size_t App::remaining_size(bool recurse) const {
  auto remaining_options = static_cast<std::size_t>(std::count_if(
      std::begin(missing_), std::end(missing_), [](const std::pair<detail::Classifier, std::string> &val) {
        return val.first != detail::Classifier::POSITIONAL_MARK;
      }));

  if (recurse) {
    for (const App_p &sub : subcommands_) {
      remaining_options += sub->remaining_size(recurse);
    }
  }
  return remaining_options;
}

CLI11_INLINE void App::_validate() const {
  // count the number of positional only args
  auto pcount = std::count_if(std::begin(options_), std::end(options_), [](const Option_p &opt) {
    return opt->get_items_expected_max() >= detail::expected_max_vector_size && !opt->nonpositional();
  });
  if (pcount > 1) {
    auto pcount_req = std::count_if(std::begin(options_), std::end(options_), [](const Option_p &opt) {
      return opt->get_items_expected_max() >= detail::expected_max_vector_size && !opt->nonpositional() &&
             opt->get_required();
    });
    if (pcount - pcount_req > 1) {
      throw InvalidError(name_);
    }
  }

  std::size_t nameless_subs{0};
  for (const App_p &app : subcommands_) {
    app->_validate();
    if (app->get_name().empty())
      ++nameless_subs;
  }

  if (require_option_min_ > 0) {
    if (require_option_max_ > 0) {
      if (require_option_max_ < require_option_min_) {
        throw(InvalidError("Required min options greater than required max options", ExitCodes::InvalidError));
      }
    }
    if (require_option_min_ > (options_.size() + nameless_subs)) {
      throw(
          InvalidError("Required min options greater than number of available options", ExitCodes::InvalidError));
    }
  }
}

CLI11_INLINE void App::_configure() {
  if (default_startup == startup_mode::enabled) {
    disabled_ = false;
  } else if (default_startup == startup_mode::disabled) {
    disabled_ = true;
  }
  for (const App_p &app : subcommands_) {
    if (app->has_automatic_name_) {
      app->name_.clear();
    }
    if (app->name_.empty()) {
      app->fallthrough_ = false; // make sure fallthrough_ is false to prevent infinite loop
      app->prefix_command_ = false;
    }
    // make sure the parent is set to be this object in preparation for parse
    app->parent_ = this;
    app->_configure();
  }
}

CLI11_INLINE void App::run_callback(bool final_mode, bool suppress_final_callback) {
  pre_callback();
  // in the main app if immediate_callback_ is set it runs the main callback before the used subcommands
  if (!final_mode && parse_complete_callback_) {
    parse_complete_callback_();
  }
  // run the callbacks for the received subcommands
  for (App *subc : get_subcommands()) {
    if (subc->parent_ == this) {
      subc->run_callback(true, suppress_final_callback);
    }
  }
  // now run callbacks for option_groups
  for (auto &subc : subcommands_) {
    if (subc->name_.empty() && subc->count_all() > 0) {
      subc->run_callback(true, suppress_final_callback);
    }
  }

  // finally run the main callback
  if (final_callback_ && (parsed_ > 0) && (!suppress_final_callback)) {
    if (!name_.empty() || count_all() > 0 || parent_ == nullptr) {
      final_callback_();
    }
  }
}

CLI11_NODISCARD CLI11_INLINE bool App::_valid_subcommand(const std::string &current, bool ignore_used) const {
  // Don't match if max has been reached - but still check parents
  if (require_subcommand_max_ != 0 && parsed_subcommands_.size() >= require_subcommand_max_) {
    return parent_ != nullptr && parent_->_valid_subcommand(current, ignore_used);
  }
  auto *com = _find_subcommand(current, true, ignore_used);
  if (com != nullptr) {
    return true;
  }
  // Check parent if exists, else return false
  return parent_ != nullptr && parent_->_valid_subcommand(current, ignore_used);
}

CLI11_NODISCARD CLI11_INLINE detail::Classifier App::_recognize(const std::string &current,
                                                                bool ignore_used_subcommands) const {
  std::string dummy1, dummy2;

  if (current == "--")
    return detail::Classifier::POSITIONAL_MARK;
  if (_valid_subcommand(current, ignore_used_subcommands))
    return detail::Classifier::SUBCOMMAND;
  if (detail::split_long(current, dummy1, dummy2))
    return detail::Classifier::LONG;
  if (detail::split_short(current, dummy1, dummy2)) {
    if (dummy1[0] >= '0' && dummy1[0] <= '9') {
      if (get_option_no_throw(std::string{'-', dummy1[0]}) == nullptr) {
        return detail::Classifier::NONE;
      }
    }
    return detail::Classifier::SHORT;
  }
  if ((allow_windows_style_options_) && (detail::split_windows_style(current, dummy1, dummy2)))
    return detail::Classifier::WINDOWS_STYLE;
  if ((current == "++") && !name_.empty() && parent_ != nullptr)
    return detail::Classifier::SUBCOMMAND_TERMINATOR;
  auto dotloc = current.find_first_of('.');
  if (dotloc != std::string::npos) {
    auto *cm = _find_subcommand(current.substr(0, dotloc), true, ignore_used_subcommands);
    if (cm != nullptr) {
      auto res = cm->_recognize(current.substr(dotloc + 1), ignore_used_subcommands);
      if (res == detail::Classifier::SUBCOMMAND) {
        return res;
      }
    }
  }
  return detail::Classifier::NONE;
}

CLI11_INLINE bool App::_process_config_file(const std::string &config_file, bool throw_error) {
  auto path_result = detail::check_path(config_file.c_str());
  if (path_result == detail::path_type::file) {
    try {
      std::vector<ConfigItem> values = config_formatter_->from_file(config_file);
      _parse_config(values);
      return true;
    } catch (const FileError &) {
      if (throw_error) {
        throw;
      }
      return false;
    }
  } else if (throw_error) {
    throw FileError::Missing(config_file);
  } else {
    return false;
  }
}

CLI11_INLINE void App::_process_config_file() {
  if (config_ptr_ != nullptr) {
    bool config_required = config_ptr_->get_required();
    auto file_given = config_ptr_->count() > 0;
    if (!(file_given || config_ptr_->envname_.empty())) {
      std::string ename_string = detail::get_environment_value(config_ptr_->envname_);
      if (!ename_string.empty()) {
        config_ptr_->add_result(ename_string);
      }
    }
    config_ptr_->run_callback();

    auto config_files = config_ptr_->as<std::vector<std::string>>();
    bool files_used{file_given};
    if (config_files.empty() || config_files.front().empty()) {
      if (config_required) {
        throw FileError("config file is required but none was given");
      }
      return;
    }
    for (const auto &config_file : config_files) {
      if (_process_config_file(config_file, config_required || file_given)) {
        files_used = true;
      }
    }
    if (!files_used) {
      // this is done so the count shows as 0 if no callbacks were processed
      config_ptr_->clear();
      bool force = config_ptr_->force_callback_;
      config_ptr_->force_callback_ = false;
      config_ptr_->run_callback();
      config_ptr_->force_callback_ = force;
    }
  }
}

CLI11_INLINE void App::_process_env() {
  for (const Option_p &opt : options_) {
    if (opt->count() == 0 && !opt->envname_.empty()) {
      std::string ename_string = detail::get_environment_value(opt->envname_);
      if (!ename_string.empty()) {
        std::string result = ename_string;
        result = opt->_validate(result, 0);
        if (result.empty()) {
          opt->add_result(ename_string);
        }
      }
    }
  }

  for (App_p &sub : subcommands_) {
    if (sub->get_name().empty() || (sub->count_all() > 0 && !sub->parse_complete_callback_)) {
      // only process environment variables if the callback has actually been triggered already
      sub->_process_env();
    }
  }
}

CLI11_INLINE void App::_process_callbacks() {

  for (App_p &sub : subcommands_) {
    // process the priority option_groups first
    if (sub->get_name().empty() && sub->parse_complete_callback_) {
      if (sub->count_all() > 0) {
        sub->_process_callbacks();
        sub->run_callback();
      }
    }
  }

  for (const Option_p &opt : options_) {
    if ((*opt) && !opt->get_callback_run()) {
      opt->run_callback();
    }
  }
  for (App_p &sub : subcommands_) {
    if (!sub->parse_complete_callback_) {
      sub->_process_callbacks();
    }
  }
}

CLI11_INLINE void App::_process_help_flags(bool trigger_help, bool trigger_all_help) const {
  const Option *help_ptr = get_help_ptr();
  const Option *help_all_ptr = get_help_all_ptr();

  if (help_ptr != nullptr && help_ptr->count() > 0)
    trigger_help = true;
  if (help_all_ptr != nullptr && help_all_ptr->count() > 0)
    trigger_all_help = true;

  // If there were parsed subcommands, call those. First subcommand wins if there are multiple ones.
  if (!parsed_subcommands_.empty()) {
    for (const App *sub : parsed_subcommands_)
      sub->_process_help_flags(trigger_help, trigger_all_help);

    // Only the final subcommand should call for help. All help wins over help.
  } else if (trigger_all_help) {
    throw CallForAllHelp();
  } else if (trigger_help) {
    throw CallForHelp();
  }
}

CLI11_INLINE void App::_process_requirements() {
  // check excludes
  bool excluded{false};
  std::string excluder;
  for (const auto &opt : exclude_options_) {
    if (opt->count() > 0) {
      excluded = true;
      excluder = opt->get_name();
    }
  }
  for (const auto &subc : exclude_subcommands_) {
    if (subc->count_all() > 0) {
      excluded = true;
      excluder = subc->get_display_name();
    }
  }
  if (excluded) {
    if (count_all() > 0) {
      throw ExcludesError(get_display_name(), excluder);
    }
    // if we are excluded but didn't receive anything, just return
    return;
  }

  // check excludes
  bool missing_needed{false};
  std::string missing_need;
  for (const auto &opt : need_options_) {
    if (opt->count() == 0) {
      missing_needed = true;
      missing_need = opt->get_name();
    }
  }
  for (const auto &subc : need_subcommands_) {
    if (subc->count_all() == 0) {
      missing_needed = true;
      missing_need = subc->get_display_name();
    }
  }
  if (missing_needed) {
    if (count_all() > 0) {
      throw RequiresError(get_display_name(), missing_need);
    }
    // if we missing something but didn't have any options, just return
    return;
  }

  std::size_t used_options = 0;
  for (const Option_p &opt : options_) {

    if (opt->count() != 0) {
      ++used_options;
    }
    // Required but empty
    if (opt->get_required() && opt->count() == 0) {
      throw RequiredError(opt->get_name());
    }
    // Requires
    for (const Option *opt_req : opt->needs_)
      if (opt->count() > 0 && opt_req->count() == 0)
        throw RequiresError(opt->get_name(), opt_req->get_name());
    // Excludes
    for (const Option *opt_ex : opt->excludes_)
      if (opt->count() > 0 && opt_ex->count() != 0)
        throw ExcludesError(opt->get_name(), opt_ex->get_name());
  }
  // check for the required number of subcommands
  if (require_subcommand_min_ > 0) {
    auto selected_subcommands = get_subcommands();
    if (require_subcommand_min_ > selected_subcommands.size())
      throw RequiredError::Subcommand(require_subcommand_min_);
  }

  // Max error cannot occur, the extra subcommand will parse as an ExtrasError or a remaining item.

  // run this loop to check how many unnamed subcommands were actually used since they are considered options
  // from the perspective of an App
  for (App_p &sub : subcommands_) {
    if (sub->disabled_)
      continue;
    if (sub->name_.empty() && sub->count_all() > 0) {
      ++used_options;
    }
  }

  if (require_option_min_ > used_options || (require_option_max_ > 0 && require_option_max_ < used_options)) {
    auto option_list = detail::join(options_, [this](const Option_p &ptr) {
      if (ptr.get() == help_ptr_ || ptr.get() == help_all_ptr_) {
        return std::string{};
      }
      return ptr->get_name(false, true);
    });

    auto subc_list = get_subcommands([](App *app) { return ((app->get_name().empty()) && (!app->disabled_)); });
    if (!subc_list.empty()) {
      option_list += "," + detail::join(subc_list, [](const App *app) { return app->get_display_name(); });
    }
    throw RequiredError::Option(require_option_min_, require_option_max_, used_options, option_list);
  }

  // now process the requirements for subcommands if needed
  for (App_p &sub : subcommands_) {
    if (sub->disabled_)
      continue;
    if (sub->name_.empty() && sub->required_ == false) {
      if (sub->count_all() == 0) {
        if (require_option_min_ > 0 && require_option_min_ <= used_options) {
          continue;
          // if we have met the requirement and there is nothing in this option group skip checking
          // requirements
        }
        if (require_option_max_ > 0 && used_options >= require_option_min_) {
          continue;
          // if we have met the requirement and there is nothing in this option group skip checking
          // requirements
        }
      }
    }
    if (sub->count() > 0 || sub->name_.empty()) {
      sub->_process_requirements();
    }

    if (sub->required_ && sub->count_all() == 0) {
      throw(CLI::RequiredError(sub->get_display_name()));
    }
  }
}

CLI11_INLINE void App::_process() {
  try {
    // the config file might generate a FileError but that should not be processed until later in the process
    // to allow for help, version and other errors to generate first.
    _process_config_file();

    // process env shouldn't throw but no reason to process it if config generated an error
    _process_env();
  } catch (const CLI::FileError &) {
    // callbacks and help_flags can generate exceptions which should take priority
    // over the config file error if one exists.
    _process_callbacks();
    _process_help_flags();
    throw;
  }

  _process_callbacks();
  _process_help_flags();

  _process_requirements();
}

CLI11_INLINE void App::_process_extras() {
  if (!(allow_extras_ || prefix_command_)) {
    std::size_t num_left_over = remaining_size();
    if (num_left_over > 0) {
      throw ExtrasError(name_, remaining(false));
    }
  }

  for (App_p &sub : subcommands_) {
    if (sub->count() > 0)
      sub->_process_extras();
  }
}

CLI11_INLINE void App::_process_extras(std::vector<std::string> &args) {
  if (!(allow_extras_ || prefix_command_)) {
    std::size_t num_left_over = remaining_size();
    if (num_left_over > 0) {
      args = remaining(false);
      throw ExtrasError(name_, args);
    }
  }

  for (App_p &sub : subcommands_) {
    if (sub->count() > 0)
      sub->_process_extras(args);
  }
}

CLI11_INLINE void App::increment_parsed() {
  ++parsed_;
  for (App_p &sub : subcommands_) {
    if (sub->get_name().empty())
      sub->increment_parsed();
  }
}

CLI11_INLINE void App::_parse(std::vector<std::string> &args) {
  increment_parsed();
  _trigger_pre_parse(args.size());
  bool positional_only = false;

  while (!args.empty()) {
    if (!_parse_single(args, positional_only)) {
      break;
    }
  }

  if (parent_ == nullptr) {
    _process();

    // Throw error if any items are left over (depending on settings)
    _process_extras(args);

    // Convert missing (pairs) to extras (string only) ready for processing in another app
    args = remaining_for_passthrough(false);
  } else if (parse_complete_callback_) {
    _process_env();
    _process_callbacks();
    _process_help_flags();
    _process_requirements();
    run_callback(false, true);
  }
}

CLI11_INLINE void App::_parse(std::vector<std::string> &&args) {
  // this can only be called by the top level in which case parent == nullptr by definition
  // operation is simplified
  increment_parsed();
  _trigger_pre_parse(args.size());
  bool positional_only = false;

  while (!args.empty()) {
    _parse_single(args, positional_only);
  }
  _process();

  // Throw error if any items are left over (depending on settings)
  _process_extras();
}

CLI11_INLINE void App::_parse_stream(std::istream &input) {
  auto values = config_formatter_->from_config(input);
  _parse_config(values);
  increment_parsed();
  _trigger_pre_parse(values.size());
  _process();

  // Throw error if any items are left over (depending on settings)
  _process_extras();
}

CLI11_INLINE void App::_parse_config(const std::vector<ConfigItem> &args) {
  for (const ConfigItem &item : args) {
    if (!_parse_single_config(item) && allow_config_extras_ == config_extras_mode::error)
      throw ConfigError::Extras(item.fullname());
  }
}

CLI11_INLINE bool App::_parse_single_config(const ConfigItem &item, std::size_t level) {

  if (level < item.parents.size()) {
    auto *subcom = get_subcommand_no_throw(item.parents.at(level));
    return (subcom != nullptr) ? subcom->_parse_single_config(item, level + 1) : false;
  }
  // check for section open
  if (item.name == "++") {
    if (configurable_) {
      increment_parsed();
      _trigger_pre_parse(2);
      if (parent_ != nullptr) {
        parent_->parsed_subcommands_.push_back(this);
      }
    }
    return true;
  }
  // check for section close
  if (item.name == "--") {
    if (configurable_ && parse_complete_callback_) {
      _process_callbacks();
      _process_requirements();
      run_callback();
    }
    return true;
  }
  Option *op = get_option_no_throw("--" + item.name);
  if (op == nullptr) {
    if (item.name.size() == 1) {
      op = get_option_no_throw("-" + item.name);
    }
    if (op == nullptr) {
      op = get_option_no_throw(item.name);
    }
  }

  if (op == nullptr) {
    // If the option was not present
    if (get_allow_config_extras() == config_extras_mode::capture) {
      // Should we worry about classifying the extras properly?
      missing_.emplace_back(detail::Classifier::NONE, item.fullname());
      for (const auto &input : item.inputs) {
        missing_.emplace_back(detail::Classifier::NONE, input);
      }
    }
    return false;
  }

  if (!op->get_configurable()) {
    if (get_allow_config_extras() == config_extras_mode::ignore_all) {
      return false;
    }
    throw ConfigError::NotConfigurable(item.fullname());
  }

  if (op->empty()) {

    if (op->get_expected_min() == 0) {
      if (item.inputs.size() <= 1) {
        // Flag parsing
        auto res = config_formatter_->to_flag(item);
        bool converted{false};
        if (op->get_disable_flag_override()) {
          auto val = detail::to_flag_value(res);
          if (val == 1) {
            res = op->get_flag_value(item.name, "{}");
            converted = true;
          }
        }

        if (!converted) {
          errno = 0;
          res = op->get_flag_value(item.name, res);
        }

        op->add_result(res);
        return true;
      }
      if (static_cast<int>(item.inputs.size()) > op->get_items_expected_max() &&
          op->get_multi_option_policy() != MultiOptionPolicy::TakeAll) {
        if (op->get_items_expected_max() > 1) {
          throw ArgumentMismatch::AtMost(item.fullname(), op->get_items_expected_max(), item.inputs.size());
        }

        if (!op->get_disable_flag_override()) {
          throw ConversionError::TooManyInputsFlag(item.fullname());
        }
        // if the disable flag override is set then we must have the flag values match a known flag value
        // this is true regardless of the output value, so an array input is possible and must be accounted for
        for (const auto &res : item.inputs) {
          bool valid_value{false};
          if (op->default_flag_values_.empty()) {
            if (res == "true" || res == "false" || res == "1" || res == "0") {
              valid_value = true;
            }
          } else {
            for (const auto &valid_res : op->default_flag_values_) {
              if (valid_res.second == res) {
                valid_value = true;
                break;
              }
            }
          }

          if (valid_value) {
            op->add_result(res);
          } else {
            throw InvalidError("invalid flag argument given");
          }
        }
        return true;
      }
    }
    op->add_result(item.inputs);
    op->run_callback();
  }

  return true;
}

CLI11_INLINE bool App::_parse_single(std::vector<std::string> &args, bool &positional_only) {
  bool retval = true;
  detail::Classifier classifier = positional_only ? detail::Classifier::NONE : _recognize(args.back());
  switch (classifier) {
  case detail::Classifier::POSITIONAL_MARK:
    args.pop_back();
    positional_only = true;
    if ((!_has_remaining_positionals()) && (parent_ != nullptr)) {
      retval = false;
    } else {
      _move_to_missing(classifier, "--");
    }
    break;
  case detail::Classifier::SUBCOMMAND_TERMINATOR:
    // treat this like a positional mark if in the parent app
    args.pop_back();
    retval = false;
    break;
  case detail::Classifier::SUBCOMMAND:
    retval = _parse_subcommand(args);
    break;
  case detail::Classifier::LONG:
  case detail::Classifier::SHORT:
  case detail::Classifier::WINDOWS_STYLE:
    // If already parsed a subcommand, don't accept options_
    retval = _parse_arg(args, classifier, false);
    break;
  case detail::Classifier::NONE:
    // Probably a positional or something for a parent (sub)command
    retval = _parse_positional(args, false);
    if (retval && positionals_at_end_) {
      positional_only = true;
    }
    break;
    // LCOV_EXCL_START
  default:
    throw HorribleError("unrecognized classifier (you should not see this!)");
    // LCOV_EXCL_STOP
  }
  return retval;
}

CLI11_NODISCARD CLI11_INLINE std::size_t App::_count_remaining_positionals(bool required_only) const {
  std::size_t retval = 0;
  for (const Option_p &opt : options_) {
    if (opt->get_positional() && (!required_only || opt->get_required())) {
      if (opt->get_items_expected_min() > 0 && static_cast<int>(opt->count()) < opt->get_items_expected_min()) {
        retval += static_cast<std::size_t>(opt->get_items_expected_min()) - opt->count();
      }
    }
  }
  return retval;
}

CLI11_NODISCARD CLI11_INLINE bool App::_has_remaining_positionals() const {
  for (const Option_p &opt : options_) {
    if (opt->get_positional() && ((static_cast<int>(opt->count()) < opt->get_items_expected_min()))) {
      return true;
    }
  }

  return false;
}

CLI11_INLINE bool App::_parse_positional(std::vector<std::string> &args, bool haltOnSubcommand) {

  const std::string &positional = args.back();
  Option *posOpt{nullptr};

  if (positionals_at_end_) {
    // deal with the case of required arguments at the end which should take precedence over other arguments
    auto arg_rem = args.size();
    auto remreq = _count_remaining_positionals(true);
    if (arg_rem <= remreq) {
      for (const Option_p &opt : options_) {
        if (opt->get_positional() && opt->required_) {
          if (static_cast<int>(opt->count()) < opt->get_items_expected_min()) {
            if (validate_positionals_) {
              std::string pos = positional;
              pos = opt->_validate(pos, 0);
              if (!pos.empty()) {
                continue;
              }
            }
            posOpt = opt.get();
            break;
          }
        }
      }
    }
  }
  if (posOpt == nullptr) {
    for (const Option_p &opt : options_) {
      // Eat options, one by one, until done
      if (opt->get_positional() &&
          (static_cast<int>(opt->count()) < opt->get_items_expected_min() || opt->get_allow_extra_args())) {
        if (validate_positionals_) {
          std::string pos = positional;
          pos = opt->_validate(pos, 0);
          if (!pos.empty()) {
            continue;
          }
        }
        posOpt = opt.get();
        break;
      }
    }
  }
  if (posOpt != nullptr) {
    parse_order_.push_back(posOpt);
    if (posOpt->get_inject_separator()) {
      if (!posOpt->results().empty() && !posOpt->results().back().empty()) {
        posOpt->add_result(std::string{});
      }
    }
    if (posOpt->get_trigger_on_parse() && posOpt->current_option_state_ == Option::option_state::callback_run) {
      posOpt->clear();
    }
    posOpt->add_result(positional);
    if (posOpt->get_trigger_on_parse()) {
      posOpt->run_callback();
    }

    args.pop_back();
    return true;
  }

  for (auto &subc : subcommands_) {
    if ((subc->name_.empty()) && (!subc->disabled_)) {
      if (subc->_parse_positional(args, false)) {
        if (!subc->pre_parse_called_) {
          subc->_trigger_pre_parse(args.size());
        }
        return true;
      }
    }
  }
  // let the parent deal with it if possible
  if (parent_ != nullptr && fallthrough_)
    return _get_fallthrough_parent()->_parse_positional(args, static_cast<bool>(parse_complete_callback_));

  /// Try to find a local subcommand that is repeated
  auto *com = _find_subcommand(args.back(), true, false);
  if (com != nullptr && (require_subcommand_max_ == 0 || require_subcommand_max_ > parsed_subcommands_.size())) {
    if (haltOnSubcommand) {
      return false;
    }
    args.pop_back();
    com->_parse(args);
    return true;
  }
  /// now try one last gasp at subcommands that have been executed before, go to root app and try to find a
  /// subcommand in a broader way, if one exists let the parent deal with it
  auto *parent_app = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
  com = parent_app->_find_subcommand(args.back(), true, false);
  if (com != nullptr && (com->parent_->require_subcommand_max_ == 0 ||
                         com->parent_->require_subcommand_max_ > com->parent_->parsed_subcommands_.size())) {
    return false;
  }

  if (positionals_at_end_) {
    throw CLI::ExtrasError(name_, args);
  }
  /// If this is an option group don't deal with it
  if (parent_ != nullptr && name_.empty()) {
    return false;
  }
  /// We are out of other options this goes to missing
  _move_to_missing(detail::Classifier::NONE, positional);
  args.pop_back();
  if (prefix_command_) {
    while (!args.empty()) {
      _move_to_missing(detail::Classifier::NONE, args.back());
      args.pop_back();
    }
  }

  return true;
}

CLI11_NODISCARD CLI11_INLINE App *
App::_find_subcommand(const std::string &subc_name, bool ignore_disabled, bool ignore_used) const noexcept {
  for (const App_p &com : subcommands_) {
    if (com->disabled_ && ignore_disabled)
      continue;
    if (com->get_name().empty()) {
      auto *subc = com->_find_subcommand(subc_name, ignore_disabled, ignore_used);
      if (subc != nullptr) {
        return subc;
      }
    }
    if (com->check_name(subc_name)) {
      if ((!*com) || !ignore_used)
        return com.get();
    }
  }
  return nullptr;
}

CLI11_INLINE bool App::_parse_subcommand(std::vector<std::string> &args) {
  if (_count_remaining_positionals(/* required */ true) > 0) {
    _parse_positional(args, false);
    return true;
  }
  auto *com = _find_subcommand(args.back(), true, true);
  if (com == nullptr) {
    // the main way to get here is using .notation
    auto dotloc = args.back().find_first_of('.');
    if (dotloc != std::string::npos) {
      com = _find_subcommand(args.back().substr(0, dotloc), true, true);
      if (com != nullptr) {
        args.back() = args.back().substr(dotloc + 1);
        args.push_back(com->get_display_name());
      }
    }
  }
  if (com != nullptr) {
    args.pop_back();
    if (!com->silent_) {
      parsed_subcommands_.push_back(com);
    }
    com->_parse(args);
    auto *parent_app = com->parent_;
    while (parent_app != this) {
      parent_app->_trigger_pre_parse(args.size());
      if (!com->silent_) {
        parent_app->parsed_subcommands_.push_back(com);
      }
      parent_app = parent_app->parent_;
    }
    return true;
  }

  if (parent_ == nullptr)
    throw HorribleError("Subcommand " + args.back() + " missing");
  return false;
}

CLI11_INLINE bool
App::_parse_arg(std::vector<std::string> &args, detail::Classifier current_type, bool local_processing_only) {

  std::string current = args.back();

  std::string arg_name;
  std::string value;
  std::string rest;

  switch (current_type) {
  case detail::Classifier::LONG:
    if (!detail::split_long(current, arg_name, value))
      throw HorribleError("Long parsed but missing (you should not see this):" + args.back());
    break;
  case detail::Classifier::SHORT:
    if (!detail::split_short(current, arg_name, rest))
      throw HorribleError("Short parsed but missing! You should not see this");
    break;
  case detail::Classifier::WINDOWS_STYLE:
    if (!detail::split_windows_style(current, arg_name, value))
      throw HorribleError("windows option parsed but missing! You should not see this");
    break;
  case detail::Classifier::SUBCOMMAND:
  case detail::Classifier::SUBCOMMAND_TERMINATOR:
  case detail::Classifier::POSITIONAL_MARK:
  case detail::Classifier::NONE:
  default:
    throw HorribleError("parsing got called with invalid option! You should not see this");
  }

  auto op_ptr = std::find_if(std::begin(options_), std::end(options_), [arg_name, current_type](const Option_p &opt) {
    if (current_type == detail::Classifier::LONG)
      return opt->check_lname(arg_name);
    if (current_type == detail::Classifier::SHORT)
      return opt->check_sname(arg_name);
    // this will only get called for detail::Classifier::WINDOWS_STYLE
    return opt->check_lname(arg_name) || opt->check_sname(arg_name);
  });

  // Option not found
  if (op_ptr == std::end(options_)) {
    for (auto &subc : subcommands_) {
      if (subc->name_.empty() && !subc->disabled_) {
        if (subc->_parse_arg(args, current_type, local_processing_only)) {
          if (!subc->pre_parse_called_) {
            subc->_trigger_pre_parse(args.size());
          }
          return true;
        }
      }
    }

    // don't capture missing if this is a nameless subcommand and nameless subcommands can't fallthrough
    if (parent_ != nullptr && name_.empty()) {
      return false;
    }

    // now check for '.' notation of subcommands
    auto dotloc = arg_name.find_first_of('.', 1);
    if (dotloc != std::string::npos) {
      // using dot notation is equivalent to single argument subcommand
      auto *sub = _find_subcommand(arg_name.substr(0, dotloc), true, false);
      if (sub != nullptr) {
        auto v = args.back();
        args.pop_back();
        arg_name = arg_name.substr(dotloc + 1);
        if (arg_name.size() > 1) {
          args.push_back(std::string("--") + v.substr(dotloc + 3));
          current_type = detail::Classifier::LONG;
        } else {
          auto nval = v.substr(dotloc + 2);
          nval.front() = '-';
          if (nval.size() > 2) {
            // '=' not allowed in short form arguments
            args.push_back(nval.substr(3));
            nval.resize(2);
          }
          args.push_back(nval);
          current_type = detail::Classifier::SHORT;
        }
        auto val = sub->_parse_arg(args, current_type, true);
        if (val) {
          if (!sub->silent_) {
            parsed_subcommands_.push_back(sub);
          }
          // deal with preparsing
          increment_parsed();
          _trigger_pre_parse(args.size());
          // run the parse complete callback since the subcommand processing is now complete
          if (sub->parse_complete_callback_) {
            sub->_process_env();
            sub->_process_callbacks();
            sub->_process_help_flags();
            sub->_process_requirements();
            sub->run_callback(false, true);
          }
          return true;
        }
        args.pop_back();
        args.push_back(v);
      }
    }
    if (local_processing_only) {
      return false;
    }
    // If a subcommand, try the main command
    if (parent_ != nullptr && fallthrough_)
      return _get_fallthrough_parent()->_parse_arg(args, current_type, false);

    // Otherwise, add to missing
    args.pop_back();
    _move_to_missing(current_type, current);
    return true;
  }

  args.pop_back();

  // Get a reference to the pointer to make syntax bearable
  Option_p &op = *op_ptr;
  /// if we require a separator add it here
  if (op->get_inject_separator()) {
    if (!op->results().empty() && !op->results().back().empty()) {
      op->add_result(std::string{});
    }
  }
  if (op->get_trigger_on_parse() && op->current_option_state_ == Option::option_state::callback_run) {
    op->clear();
  }
  int min_num = (std::min)(op->get_type_size_min(), op->get_items_expected_min());
  int max_num = op->get_items_expected_max();
  // check container like options to limit the argument size to a single type if the allow_extra_flags argument is
  // set. 16 is somewhat arbitrary (needs to be at least 4)
  if (max_num >= detail::expected_max_vector_size / 16 && !op->get_allow_extra_args()) {
    auto tmax = op->get_type_size_max();
    max_num = detail::checked_multiply(tmax, op->get_expected_min()) ? tmax : detail::expected_max_vector_size;
  }
  // Make sure we always eat the minimum for unlimited vectors
  int collected = 0;    // total number of arguments collected
  int result_count = 0; // local variable for number of results in a single arg string
  // deal with purely flag like things
  if (max_num == 0) {
    auto res = op->get_flag_value(arg_name, value);
    op->add_result(res);
    parse_order_.push_back(op.get());
  } else if (!value.empty()) { // --this=value
    op->add_result(value, result_count);
    parse_order_.push_back(op.get());
    collected += result_count;
    // -Trest
  } else if (!rest.empty()) {
    op->add_result(rest, result_count);
    parse_order_.push_back(op.get());
    rest = "";
    collected += result_count;
  }

  // gather the minimum number of arguments
  while (min_num > collected && !args.empty()) {
    std::string current_ = args.back();
    args.pop_back();
    op->add_result(current_, result_count);
    parse_order_.push_back(op.get());
    collected += result_count;
  }

  if (min_num > collected) { // if we have run out of arguments and the minimum was not met
    throw ArgumentMismatch::TypedAtLeast(op->get_name(), min_num, op->get_type_name());
  }

  // now check for optional arguments
  if (max_num > collected || op->get_allow_extra_args()) { // we allow optional arguments
    auto remreqpos = _count_remaining_positionals(true);
    // we have met the minimum now optionally check up to the maximum
    while ((collected < max_num || op->get_allow_extra_args()) && !args.empty() &&
           _recognize(args.back(), false) == detail::Classifier::NONE) {
      // If any required positionals remain, don't keep eating
      if (remreqpos >= args.size()) {
        break;
      }
      if (validate_optional_arguments_) {
        std::string arg = args.back();
        arg = op->_validate(arg, 0);
        if (!arg.empty()) {
          break;
        }
      }
      op->add_result(args.back(), result_count);
      parse_order_.push_back(op.get());
      args.pop_back();
      collected += result_count;
    }

    // Allow -- to end an unlimited list and "eat" it
    if (!args.empty() && _recognize(args.back()) == detail::Classifier::POSITIONAL_MARK)
      args.pop_back();
    // optional flag that didn't receive anything now get the default value
    if (min_num == 0 && max_num > 0 && collected == 0) {
      auto res = op->get_flag_value(arg_name, std::string{});
      op->add_result(res);
      parse_order_.push_back(op.get());
    }
  }
  // if we only partially completed a type then add an empty string if allowed for later processing
  if (min_num > 0 && (collected % op->get_type_size_max()) != 0) {
    if (op->get_type_size_max() != op->get_type_size_min()) {
      op->add_result(std::string{});
    } else {
      throw ArgumentMismatch::PartialType(op->get_name(), op->get_type_size_min(), op->get_type_name());
    }
  }
  if (op->get_trigger_on_parse()) {
    op->run_callback();
  }
  if (!rest.empty()) {
    rest = "-" + rest;
    args.push_back(rest);
  }
  return true;
}

CLI11_INLINE void App::_trigger_pre_parse(std::size_t remaining_args) {
  if (!pre_parse_called_) {
    pre_parse_called_ = true;
    if (pre_parse_callback_) {
      pre_parse_callback_(remaining_args);
    }
  } else if (immediate_callback_) {
    if (!name_.empty()) {
      auto pcnt = parsed_;
      missing_t extras = std::move(missing_);
      clear();
      parsed_ = pcnt;
      pre_parse_called_ = true;
      missing_ = std::move(extras);
    }
  }
}

CLI11_INLINE App *App::_get_fallthrough_parent() {
  if (parent_ == nullptr) {
    throw(HorribleError("No Valid parent"));
  }
  auto *fallthrough_parent = parent_;
  while ((fallthrough_parent->parent_ != nullptr) && (fallthrough_parent->get_name().empty())) {
    fallthrough_parent = fallthrough_parent->parent_;
  }
  return fallthrough_parent;
}

CLI11_NODISCARD CLI11_INLINE const std::string &App::_compare_subcommand_names(const App &subcom,
                                                                               const App &base) const {
  static const std::string estring;
  if (subcom.disabled_) {
    return estring;
  }
  for (const auto &subc : base.subcommands_) {
    if (subc.get() != &subcom) {
      if (subc->disabled_) {
        continue;
      }
      if (!subcom.get_name().empty()) {
        if (subc->check_name(subcom.get_name())) {
          return subcom.get_name();
        }
      }
      if (!subc->get_name().empty()) {
        if (subcom.check_name(subc->get_name())) {
          return subc->get_name();
        }
      }
      for (const auto &les : subcom.aliases_) {
        if (subc->check_name(les)) {
          return les;
        }
      }
      // this loop is needed in case of ignore_underscore or ignore_case on one but not the other
      for (const auto &les : subc->aliases_) {
        if (subcom.check_name(les)) {
          return les;
        }
      }
      // if the subcommand is an option group we need to check deeper
      if (subc->get_name().empty()) {
        const auto &cmpres = _compare_subcommand_names(subcom, *subc);
        if (!cmpres.empty()) {
          return cmpres;
        }
      }
      // if the test subcommand is an option group we need to check deeper
      if (subcom.get_name().empty()) {
        const auto &cmpres = _compare_subcommand_names(*subc, subcom);
        if (!cmpres.empty()) {
          return cmpres;
        }
      }
    }
  }
  return estring;
}

CLI11_INLINE void App::_move_to_missing(detail::Classifier val_type, const std::string &val) {
  if (allow_extras_ || subcommands_.empty()) {
    missing_.emplace_back(val_type, val);
    return;
  }
  // allow extra arguments to be places in an option group if it is allowed there
  for (auto &subc : subcommands_) {
    if (subc->name_.empty() && subc->allow_extras_) {
      subc->missing_.emplace_back(val_type, val);
      return;
    }
  }
  // if we haven't found any place to put them yet put them in missing
  missing_.emplace_back(val_type, val);
}

CLI11_INLINE void App::_move_option(Option *opt, App *app) {
  if (opt == nullptr) {
    throw OptionNotFound("the option is NULL");
  }
  // verify that the give app is actually a subcommand
  bool found = false;
  for (auto &subc : subcommands_) {
    if (app == subc.get()) {
      found = true;
    }
  }
  if (!found) {
    throw OptionNotFound("The Given app is not a subcommand");
  }

  if ((help_ptr_ == opt) || (help_all_ptr_ == opt))
    throw OptionAlreadyAdded("cannot move help options");

  if (config_ptr_ == opt)
    throw OptionAlreadyAdded("cannot move config file options");

  auto iterator =
      std::find_if(std::begin(options_), std::end(options_), [opt](const Option_p &v) { return v.get() == opt; });
  if (iterator != std::end(options_)) {
    const auto &opt_p = *iterator;
    if (std::find_if(std::begin(app->options_), std::end(app->options_), [&opt_p](const Option_p &v) {
          return (*v == *opt_p);
        }) == std::end(app->options_)) {
      // only erase after the insertion was successful
      app->options_.push_back(std::move(*iterator));
      options_.erase(iterator);
    } else {
      throw OptionAlreadyAdded("option was not located: " + opt->get_name());
    }
  } else {
    throw OptionNotFound("could not locate the given Option");
  }
}

CLI11_INLINE void TriggerOn(App *trigger_app, App *app_to_enable) {
  app_to_enable->enabled_by_default(false);
  app_to_enable->disabled_by_default();
  trigger_app->preparse_callback([app_to_enable](std::size_t) { app_to_enable->disabled(false); });
}

CLI11_INLINE void TriggerOn(App *trigger_app, std::vector<App *> apps_to_enable) {
  for (auto &app : apps_to_enable) {
    app->enabled_by_default(false);
    app->disabled_by_default();
  }

  trigger_app->preparse_callback([apps_to_enable](std::size_t) {
    for (const auto &app : apps_to_enable) {
      app->disabled(false);
    }
  });
}

CLI11_INLINE void TriggerOff(App *trigger_app, App *app_to_enable) {
  app_to_enable->disabled_by_default(false);
  app_to_enable->enabled_by_default();
  trigger_app->preparse_callback([app_to_enable](std::size_t) { app_to_enable->disabled(); });
}

CLI11_INLINE void TriggerOff(App *trigger_app, std::vector<App *> apps_to_enable) {
  for (auto &app : apps_to_enable) {
    app->disabled_by_default(false);
    app->enabled_by_default();
  }

  trigger_app->preparse_callback([apps_to_enable](std::size_t) {
    for (const auto &app : apps_to_enable) {
      app->disabled();
    }
  });
}

CLI11_INLINE void deprecate_option(Option *opt, const std::string &replacement) {
  Validator deprecate_warning{[opt, replacement](std::string &) {
                                std::cout << opt->get_name() << " is deprecated please use '" << replacement
                                          << "' instead\n";
                                return std::string();
                              },
                              "DEPRECATED"};
  deprecate_warning.application_index(0);
  opt->check(deprecate_warning);
  if (!replacement.empty()) {
    opt->description(opt->get_description() + " DEPRECATED: please use '" + replacement + "' instead");
  }
}

CLI11_INLINE void retire_option(App *app, Option *opt) {
  App temp;
  auto *option_copy = temp.add_option(opt->get_name(false, true))
                          ->type_size(opt->get_type_size_min(), opt->get_type_size_max())
                          ->expected(opt->get_expected_min(), opt->get_expected_max())
                          ->allow_extra_args(opt->get_allow_extra_args());

  app->remove_option(opt);
  auto *opt2 = app->add_option(option_copy->get_name(false, true), "option has been retired and has no effect");
  opt2->type_name("RETIRED")
      ->default_str("RETIRED")
      ->type_size(option_copy->get_type_size_min(), option_copy->get_type_size_max())
      ->expected(option_copy->get_expected_min(), option_copy->get_expected_max())
      ->allow_extra_args(option_copy->get_allow_extra_args());

  Validator retired_warning{[opt2](std::string &) {
                              std::cout << "WARNING " << opt2->get_name() << " is retired and has no effect\n";
                              return std::string();
                            },
                            ""};
  retired_warning.application_index(0);
  opt2->check(retired_warning);
}

CLI11_INLINE void retire_option(App &app, Option *opt) { retire_option(&app, opt); }

CLI11_INLINE void retire_option(App *app, const std::string &option_name) {

  auto *opt = app->get_option_no_throw(option_name);
  if (opt != nullptr) {
    retire_option(app, opt);
    return;
  }
  auto *opt2 = app->add_option(option_name, "option has been retired and has no effect")
                   ->type_name("RETIRED")
                   ->expected(0, 1)
                   ->default_str("RETIRED");
  Validator retired_warning{[opt2](std::string &) {
                              std::cout << "WARNING " << opt2->get_name() << " is retired and has no effect\n";
                              return std::string();
                            },
                            ""};
  retired_warning.application_index(0);
  opt2->check(retired_warning);
}

CLI11_INLINE void retire_option(App &app, const std::string &option_name) { retire_option(&app, option_name); }

namespace FailureMessage {

CLI11_INLINE std::string simple(const App *app, const Error &e) {
  std::string header = std::string(e.what()) + "\n";
  std::vector<std::string> names;

  // Collect names
  if (app->get_help_ptr() != nullptr)
    names.push_back(app->get_help_ptr()->get_name());

  if (app->get_help_all_ptr() != nullptr)
    names.push_back(app->get_help_all_ptr()->get_name());

  // If any names found, suggest those
  if (!names.empty())
    header += "Run with " + detail::join(names, " or ") + " for more information.\n";

  return header;
}

CLI11_INLINE std::string help(const App *app, const Error &e) {
  std::string header = std::string("ERROR: ") + e.get_name() + ": " + e.what() + "\n";
  header += app->help();
  return header;
}

} // namespace FailureMessage

namespace detail {

std::string convert_arg_for_ini(const std::string &arg,
                                char stringQuote = '"',
                                char literalQuote = '\'',
                                bool disable_multi_line = false);

/// Comma separated join, adds quotes if needed
std::string ini_join(const std::vector<std::string> &args,
                     char sepChar = ',',
                     char arrayStart = '[',
                     char arrayEnd = ']',
                     char stringQuote = '"',
                     char literalQuote = '\'');

void clean_name_string(std::string &name, const std::string &keyChars);

std::vector<std::string> generate_parents(const std::string &section, std::string &name, char parentSeparator);

/// assuming non default segments do a check on the close and open of the segments in a configItem structure
void checkParentSegments(std::vector<ConfigItem> &output, const std::string &currentSection, char parentSeparator);
} // namespace detail

static constexpr auto multiline_literal_quote = R"(''')";
static constexpr auto multiline_string_quote = R"(""")";

namespace detail {

CLI11_INLINE bool is_printable(const std::string &test_string) {
  return std::all_of(test_string.begin(), test_string.end(), [](char x) {
    return (isprint(static_cast<unsigned char>(x)) != 0 || x == '\n' || x == '\t');
  });
}

CLI11_INLINE std::string
convert_arg_for_ini(const std::string &arg, char stringQuote, char literalQuote, bool disable_multi_line) {
  if (arg.empty()) {
    return std::string(2, stringQuote);
  }
  // some specifically supported strings
  if (arg == "true" || arg == "false" || arg == "nan" || arg == "inf") {
    return arg;
  }
  // floating point conversion can convert some hex codes, but don't try that here
  if (arg.compare(0, 2, "0x") != 0 && arg.compare(0, 2, "0X") != 0) {
    using CLI::detail::lexical_cast;
    double val = 0.0;
    if (lexical_cast(arg, val)) {
      if (arg.find_first_not_of("0123456789.-+eE") == std::string::npos) {
        return arg;
      }
    }
  }
  // just quote a single non numeric character
  if (arg.size() == 1) {
    if (isprint(static_cast<unsigned char>(arg.front())) == 0) {
      return binary_escape_string(arg);
    }
    if (arg == "'") {
      return std::string(1, stringQuote) + "'" + stringQuote;
    }
    return std::string(1, literalQuote) + arg + literalQuote;
  }
  // handle hex, binary or octal arguments
  if (arg.front() == '0') {
    if (arg[1] == 'x') {
      if (std::all_of(arg.begin() + 2, arg.end(), [](char x) {
            return (x >= '0' && x <= '9') || (x >= 'A' && x <= 'F') || (x >= 'a' && x <= 'f');
          })) {
        return arg;
      }
    } else if (arg[1] == 'o') {
      if (std::all_of(arg.begin() + 2, arg.end(), [](char x) { return (x >= '0' && x <= '7'); })) {
        return arg;
      }
    } else if (arg[1] == 'b') {
      if (std::all_of(arg.begin() + 2, arg.end(), [](char x) { return (x == '0' || x == '1'); })) {
        return arg;
      }
    }
  }
  if (!is_printable(arg)) {
    return binary_escape_string(arg);
  }
  if (detail::has_escapable_character(arg)) {
    if (arg.size() > 100 && !disable_multi_line) {
      return std::string(multiline_literal_quote) + arg + multiline_literal_quote;
    }
    return std::string(1, stringQuote) + detail::add_escaped_characters(arg) + stringQuote;
  }
  return std::string(1, stringQuote) + arg + stringQuote;
}

CLI11_INLINE std::string ini_join(const std::vector<std::string> &args,
                                  char sepChar,
                                  char arrayStart,
                                  char arrayEnd,
                                  char stringQuote,
                                  char literalQuote) {
  bool disable_multi_line{false};
  std::string joined;
  if (args.size() > 1 && arrayStart != '\0') {
    joined.push_back(arrayStart);
    disable_multi_line = true;
  }
  std::size_t start = 0;
  for (const auto &arg : args) {
    if (start++ > 0) {
      joined.push_back(sepChar);
      if (!std::isspace<char>(sepChar, std::locale())) {
        joined.push_back(' ');
      }
    }
    joined.append(convert_arg_for_ini(arg, stringQuote, literalQuote, disable_multi_line));
  }
  if (args.size() > 1 && arrayEnd != '\0') {
    joined.push_back(arrayEnd);
  }
  return joined;
}

CLI11_INLINE std::vector<std::string>
generate_parents(const std::string &section, std::string &name, char parentSeparator) {
  std::vector<std::string> parents;
  if (detail::to_lower(section) != "default") {
    if (section.find(parentSeparator) != std::string::npos) {
      parents = detail::split_up(section, parentSeparator);
    } else {
      parents = {section};
    }
  }
  if (name.find(parentSeparator) != std::string::npos) {
    std::vector<std::string> plist = detail::split_up(name, parentSeparator);
    name = plist.back();
    plist.pop_back();
    parents.insert(parents.end(), plist.begin(), plist.end());
  }
  // clean up quotes on the parents
  try {
    detail::remove_quotes(parents);
  } catch (const std::invalid_argument &iarg) {
    throw CLI::ParseError(iarg.what(), CLI::ExitCodes::InvalidError);
  }
  return parents;
}

CLI11_INLINE void
checkParentSegments(std::vector<ConfigItem> &output, const std::string &currentSection, char parentSeparator) {

  std::string estring;
  auto parents = detail::generate_parents(currentSection, estring, parentSeparator);
  if (!output.empty() && output.back().name == "--") {
    std::size_t msize = (parents.size() > 1U) ? parents.size() : 2;
    while (output.back().parents.size() >= msize) {
      output.push_back(output.back());
      output.back().parents.pop_back();
    }

    if (parents.size() > 1) {
      std::size_t common = 0;
      std::size_t mpair = (std::min)(output.back().parents.size(), parents.size() - 1);
      for (std::size_t ii = 0; ii < mpair; ++ii) {
        if (output.back().parents[ii] != parents[ii]) {
          break;
        }
        ++common;
      }
      if (common == mpair) {
        output.pop_back();
      } else {
        while (output.back().parents.size() > common + 1) {
          output.push_back(output.back());
          output.back().parents.pop_back();
        }
      }
      for (std::size_t ii = common; ii < parents.size() - 1; ++ii) {
        output.emplace_back();
        output.back().parents.assign(parents.begin(), parents.begin() + static_cast<std::ptrdiff_t>(ii) + 1);
        output.back().name = "++";
      }
    }
  } else if (parents.size() > 1) {
    for (std::size_t ii = 0; ii < parents.size() - 1; ++ii) {
      output.emplace_back();
      output.back().parents.assign(parents.begin(), parents.begin() + static_cast<std::ptrdiff_t>(ii) + 1);
      output.back().name = "++";
    }
  }

  // insert a section end which is just an empty items_buffer
  output.emplace_back();
  output.back().parents = std::move(parents);
  output.back().name = "++";
}

/// @brief  checks if a string represents a multiline comment
CLI11_INLINE bool hasMLString(std::string const &fullString, char check) {
  if (fullString.length() < 3) {
    return false;
  }
  auto it = fullString.rbegin();
  return (*it == check) && (*(it + 1) == check) && (*(it + 2) == check);
}
} // namespace detail

inline std::vector<ConfigItem> ConfigBase::from_config(std::istream &input) const {
  std::string line;
  std::string buffer;
  std::string currentSection = "default";
  std::string previousSection = "default";
  std::vector<ConfigItem> output;
  bool isDefaultArray = (arrayStart == '[' && arrayEnd == ']' && arraySeparator == ',');
  bool isINIArray = (arrayStart == '\0' || arrayStart == ' ') && arrayStart == arrayEnd;
  bool inSection{false};
  bool inMLineComment{false};
  bool inMLineValue{false};

  char aStart = (isINIArray) ? '[' : arrayStart;
  char aEnd = (isINIArray) ? ']' : arrayEnd;
  char aSep = (isINIArray && arraySeparator == ' ') ? ',' : arraySeparator;
  int currentSectionIndex{0};

  std::string line_sep_chars{parentSeparatorChar, commentChar, valueDelimiter};
  while (getline(input, buffer)) {
    std::vector<std::string> items_buffer;
    std::string name;
    line = detail::trim_copy(buffer);
    std::size_t len = line.length();
    // lines have to be at least 3 characters to have any meaning to CLI just skip the rest
    if (len < 3) {
      continue;
    }
    if (line.compare(0, 3, multiline_string_quote) == 0 || line.compare(0, 3, multiline_literal_quote) == 0) {
      inMLineComment = true;
      auto cchar = line.front();
      while (inMLineComment) {
        if (getline(input, line)) {
          detail::trim(line);
        } else {
          break;
        }
        if (detail::hasMLString(line, cchar)) {
          inMLineComment = false;
        }
      }
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      if (currentSection != "default") {
        // insert a section end which is just an empty items_buffer
        output.emplace_back();
        output.back().parents = detail::generate_parents(currentSection, name, parentSeparatorChar);
        output.back().name = "--";
      }
      currentSection = line.substr(1, len - 2);
      // deal with double brackets for TOML
      if (currentSection.size() > 1 && currentSection.front() == '[' && currentSection.back() == ']') {
        currentSection = currentSection.substr(1, currentSection.size() - 2);
      }
      if (detail::to_lower(currentSection) == "default") {
        currentSection = "default";
      } else {
        detail::checkParentSegments(output, currentSection, parentSeparatorChar);
      }
      inSection = false;
      if (currentSection == previousSection) {
        ++currentSectionIndex;
      } else {
        currentSectionIndex = 0;
        previousSection = currentSection;
      }
      continue;
    }

    // comment lines
    if (line.front() == ';' || line.front() == '#' || line.front() == commentChar) {
      continue;
    }
    std::size_t search_start = 0;
    if (line.find_first_of("\"'`") != std::string::npos) {
      while (search_start < line.size()) {
        auto test_char = line[search_start];
        if (test_char == '\"' || test_char == '\'' || test_char == '`') {
          search_start = detail::close_sequence(line, search_start, line[search_start]);
          ++search_start;
        } else if (test_char == valueDelimiter || test_char == commentChar) {
          --search_start;
          break;
        } else if (test_char == ' ' || test_char == '\t' || test_char == parentSeparatorChar) {
          ++search_start;
        } else {
          search_start = line.find_first_of(line_sep_chars, search_start);
        }
      }
    }
    // Find = in string, split and recombine
    auto delimiter_pos = line.find_first_of(valueDelimiter, search_start + 1);
    auto comment_pos = line.find_first_of(commentChar, search_start);
    if (comment_pos < delimiter_pos) {
      delimiter_pos = std::string::npos;
    }
    if (delimiter_pos != std::string::npos) {

      name = detail::trim_copy(line.substr(0, delimiter_pos));
      std::string item = detail::trim_copy(line.substr(delimiter_pos + 1, std::string::npos));
      bool mlquote =
          (item.compare(0, 3, multiline_literal_quote) == 0 || item.compare(0, 3, multiline_string_quote) == 0);
      if (!mlquote && comment_pos != std::string::npos) {
        auto citems = detail::split_up(item, commentChar);
        item = detail::trim_copy(citems.front());
      }
      if (mlquote) {
        // mutliline string
        auto keyChar = item.front();
        item = buffer.substr(delimiter_pos + 1, std::string::npos);
        detail::ltrim(item);
        item.erase(0, 3);
        inMLineValue = true;
        bool lineExtension{false};
        bool firstLine = true;
        if (!item.empty() && item.back() == '\\') {
          item.pop_back();
          lineExtension = true;
        }
        while (inMLineValue) {
          std::string l2;
          if (!std::getline(input, l2)) {
            break;
          }
          line = l2;
          detail::rtrim(line);
          if (detail::hasMLString(line, keyChar)) {
            line.pop_back();
            line.pop_back();
            line.pop_back();
            if (lineExtension) {
              detail::ltrim(line);
            } else if (!(firstLine && item.empty())) {
              item.push_back('\n');
            }
            firstLine = false;
            item += line;
            inMLineValue = false;
            if (!item.empty() && item.back() == '\n') {
              item.pop_back();
            }
            if (keyChar == '\"') {
              try {
                item = detail::remove_escaped_characters(item);
              } catch (const std::invalid_argument &iarg) {
                throw CLI::ParseError(iarg.what(), CLI::ExitCodes::InvalidError);
              }
            }
          } else {
            if (lineExtension) {
              detail::trim(l2);
            } else if (!(firstLine && item.empty())) {
              item.push_back('\n');
            }
            lineExtension = false;
            firstLine = false;
            if (!l2.empty() && l2.back() == '\\') {
              lineExtension = true;
              l2.pop_back();
            }
            item += l2;
          }
        }
        items_buffer = {item};
      } else if (item.size() > 1 && item.front() == aStart) {
        for (std::string multiline; item.back() != aEnd && std::getline(input, multiline);) {
          detail::trim(multiline);
          item += multiline;
        }
        if (item.back() == aEnd) {
          items_buffer = detail::split_up(item.substr(1, item.length() - 2), aSep);
        } else {
          items_buffer = detail::split_up(item.substr(1, std::string::npos), aSep);
        }
      } else if ((isDefaultArray || isINIArray) && item.find_first_of(aSep) != std::string::npos) {
        items_buffer = detail::split_up(item, aSep);
      } else if ((isDefaultArray || isINIArray) && item.find_first_of(' ') != std::string::npos) {
        items_buffer = detail::split_up(item, '\0');
      } else {
        items_buffer = {item};
      }
    } else {
      name = detail::trim_copy(line.substr(0, comment_pos));
      items_buffer = {"true"};
    }
    std::vector<std::string> parents;
    try {
      parents = detail::generate_parents(currentSection, name, parentSeparatorChar);
      detail::process_quoted_string(name);
      // clean up quotes on the items and check for escaped strings
      for (auto &it : items_buffer) {
        detail::process_quoted_string(it, stringQuote, literalQuote);
      }
    } catch (const std::invalid_argument &ia) {
      throw CLI::ParseError(ia.what(), CLI::ExitCodes::InvalidError);
    }

    if (parents.size() > maximumLayers) {
      continue;
    }
    if (!configSection.empty() && !inSection) {
      if (parents.empty() || parents.front() != configSection) {
        continue;
      }
      if (configIndex >= 0 && currentSectionIndex != configIndex) {
        continue;
      }
      parents.erase(parents.begin());
      inSection = true;
    }
    if (!output.empty() && name == output.back().name && parents == output.back().parents) {
      output.back().inputs.insert(output.back().inputs.end(), items_buffer.begin(), items_buffer.end());
    } else {
      output.emplace_back();
      output.back().parents = std::move(parents);
      output.back().name = std::move(name);
      output.back().inputs = std::move(items_buffer);
    }
  }
  if (currentSection != "default") {
    // insert a section end which is just an empty items_buffer
    std::string ename;
    output.emplace_back();
    output.back().parents = detail::generate_parents(currentSection, ename, parentSeparatorChar);
    output.back().name = "--";
    while (output.back().parents.size() > 1) {
      output.push_back(output.back());
      output.back().parents.pop_back();
    }
  }
  return output;
}

CLI11_INLINE std::string &clean_name_string(std::string &name, const std::string &keyChars) {
  if (name.find_first_of(keyChars) != std::string::npos || (name.front() == '[' && name.back() == ']') ||
      (name.find_first_of("'`\"\\") != std::string::npos)) {
    if (name.find_first_of('\'') == std::string::npos) {
      name.insert(0, 1, '\'');
      name.push_back('\'');
    } else {
      if (detail::has_escapable_character(name)) {
        name = detail::add_escaped_characters(name);
      }
      name.insert(0, 1, '\"');
      name.push_back('\"');
    }
  }
  return name;
}

CLI11_INLINE std::string
ConfigBase::to_config(const App *app, bool default_also, bool write_description, std::string prefix) const {
  std::stringstream out;
  std::string commentLead;
  commentLead.push_back(commentChar);
  commentLead.push_back(' ');

  std::string commentTest = "#;";
  commentTest.push_back(commentChar);
  commentTest.push_back(parentSeparatorChar);

  std::string keyChars = commentTest;
  keyChars.push_back(literalQuote);
  keyChars.push_back(stringQuote);
  keyChars.push_back(arrayStart);
  keyChars.push_back(arrayEnd);
  keyChars.push_back(valueDelimiter);
  keyChars.push_back(arraySeparator);

  std::vector<std::string> groups = app->get_groups();
  bool defaultUsed = false;
  groups.insert(groups.begin(), std::string("Options"));
  if (write_description && (app->get_configurable() || app->get_parent() == nullptr || app->get_name().empty())) {
    out << commentLead << detail::fix_newlines(commentLead, app->get_description()) << '\n';
  }
  for (auto &group : groups) {
    if (group == "Options" || group.empty()) {
      if (defaultUsed) {
        continue;
      }
      defaultUsed = true;
    }
    if (write_description && group != "Options" && !group.empty()) {
      out << '\n'
          << commentLead << group << " Options\n";
    }
    for (const Option *opt : app->get_options({})) {

      // Only process options that are configurable
      if (opt->get_configurable()) {
        if (opt->get_group() != group) {
          if (!(group == "Options" && opt->get_group().empty())) {
            continue;
          }
        }
        std::string single_name = opt->get_single_name();
        if (single_name.empty()) {
          continue;
        }

        std::string value = detail::ini_join(
            opt->reduced_results(), arraySeparator, arrayStart, arrayEnd, stringQuote, literalQuote);

        if (value.empty() && default_also) {
          if (!opt->get_default_str().empty()) {
            value = detail::convert_arg_for_ini(opt->get_default_str(), stringQuote, literalQuote, false);
          } else if (opt->get_expected_min() == 0) {
            value = "false";
          } else if (opt->get_run_callback_for_default()) {
            value = "\"\""; // empty string default value
          }
        }

        if (!value.empty()) {

          if (!opt->get_fnames().empty()) {
            try {
              value = opt->get_flag_value(single_name, value);
            } catch (const CLI::ArgumentMismatch &) {
              bool valid{false};
              for (const auto &test_name : opt->get_fnames()) {
                try {
                  value = opt->get_flag_value(test_name, value);
                  single_name = test_name;
                  valid = true;
                } catch (const CLI::ArgumentMismatch &) {
                  continue;
                }
              }
              if (!valid) {
                value = detail::ini_join(
                    opt->results(), arraySeparator, arrayStart, arrayEnd, stringQuote, literalQuote);
              }
            }
          }
          if (write_description && opt->has_description()) {
            out << '\n';
            out << commentLead << detail::fix_newlines(commentLead, opt->get_description()) << '\n';
          }
          clean_name_string(single_name, keyChars);

          std::string name = prefix + single_name;

          out << name << valueDelimiter << value << '\n';
        }
      }
    }
  }
  auto subcommands = app->get_subcommands({});
  for (const App *subcom : subcommands) {
    if (subcom->get_name().empty()) {
      if (!default_also && (subcom->count_all() == 0)) {
        continue;
      }
      if (write_description && !subcom->get_group().empty()) {
        out << '\n'
            << commentLead << subcom->get_group() << " Options\n";
      }
      /*if (!prefix.empty() || app->get_parent() == nullptr) {
          out << '[' << prefix << "___"<< subcom->get_group() << "]\n";
      } else {
          std::string subname = app->get_name() + parentSeparatorChar + "___"+subcom->get_group();
          const auto *p = app->get_parent();
          while(p->get_parent() != nullptr) {
              subname = p->get_name() + parentSeparatorChar +subname;
              p = p->get_parent();
          }
          out << '[' << subname << "]\n";
      }
      */
      out << to_config(subcom, default_also, write_description, prefix);
    }
  }

  for (const App *subcom : subcommands) {
    if (!subcom->get_name().empty()) {
      if (!default_also && (subcom->count_all() == 0)) {
        continue;
      }
      std::string subname = subcom->get_name();
      clean_name_string(subname, keyChars);

      if (subcom->get_configurable() && app->got_subcommand(subcom)) {
        if (!prefix.empty() || app->get_parent() == nullptr) {

          out << '[' << prefix << subname << "]\n";
        } else {
          std::string appname = app->get_name();
          clean_name_string(appname, keyChars);
          subname = appname + parentSeparatorChar + subname;
          const auto *p = app->get_parent();
          while (p->get_parent() != nullptr) {
            std::string pname = p->get_name();
            clean_name_string(pname, keyChars);
            subname = pname + parentSeparatorChar + subname;
            p = p->get_parent();
          }
          out << '[' << subname << "]\n";
        }
        out << to_config(subcom, default_also, write_description, "");
      } else {
        out << to_config(subcom, default_also, write_description, prefix + subname + parentSeparatorChar);
      }
    }
  }

  return out.str();
}

CLI11_INLINE std::string
Formatter::make_group(std::string group, bool is_positional, std::vector<const Option *> opts) const {
  std::stringstream out;

  out << "\n"
      << group << ":\n";
  for (const Option *opt : opts) {
    out << make_option(opt, is_positional);
  }

  return out.str();
}

CLI11_INLINE std::string Formatter::make_positionals(const App *app) const {
  std::vector<const Option *> opts =
      app->get_options([](const Option *opt) { return !opt->get_group().empty() && opt->get_positional(); });

  if (opts.empty())
    return {};

  return make_group(get_label("Positionals"), true, opts);
}

CLI11_INLINE std::string Formatter::make_groups(const App *app, AppFormatMode mode) const {
  std::stringstream out;
  std::vector<std::string> groups = app->get_groups();

  // Options
  for (const std::string &group : groups) {
    std::vector<const Option *> opts = app->get_options([app, mode, &group](const Option *opt) {
      return opt->get_group() == group                    // Must be in the right group
             && opt->nonpositional()                      // Must not be a positional
             && (mode != AppFormatMode::Sub               // If mode is Sub, then
                 || (app->get_help_ptr() != opt           // Ignore help pointer
                     && app->get_help_all_ptr() != opt)); // Ignore help all pointer
    });
    if (!group.empty() && !opts.empty()) {
      out << make_group(group, false, opts);

      if (group != groups.back())
        out << "\n";
    }
  }

  return out.str();
}

CLI11_INLINE std::string Formatter::make_description(const App *app) const {
  std::string desc = app->get_description();
  auto min_options = app->get_require_option_min();
  auto max_options = app->get_require_option_max();
  if (app->get_required()) {
    desc += " " + get_label("REQUIRED") + " ";
  }
  if ((max_options == min_options) && (min_options > 0)) {
    if (min_options == 1) {
      desc += " \n[Exactly 1 of the following options is required]";
    } else {
      desc += " \n[Exactly " + std::to_string(min_options) + " options from the following list are required]";
    }
  } else if (max_options > 0) {
    if (min_options > 0) {
      desc += " \n[Between " + std::to_string(min_options) + " and " + std::to_string(max_options) +
              " of the follow options are required]";
    } else {
      desc += " \n[At most " + std::to_string(max_options) + " of the following options are allowed]";
    }
  } else if (min_options > 0) {
    desc += " \n[At least " + std::to_string(min_options) + " of the following options are required]";
  }
  return (!desc.empty()) ? desc + "\n" : std::string{};
}

CLI11_INLINE std::string Formatter::make_usage(const App *app, std::string name) const {
  std::string usage = app->get_usage();
  if (!usage.empty()) {
    return usage + "\n";
  }

  std::stringstream out;

  out << get_label("Usage") << ":" << (name.empty() ? "" : " ") << name;

  std::vector<std::string> groups = app->get_groups();

  // Print an Options badge if any options exist
  std::vector<const Option *> non_pos_options =
      app->get_options([](const Option *opt) { return opt->nonpositional(); });
  if (!non_pos_options.empty())
    out << " [" << get_label("OPTIONS") << "]";

  // Positionals need to be listed here
  std::vector<const Option *> positionals = app->get_options([](const Option *opt) { return opt->get_positional(); });

  // Print out positionals if any are left
  if (!positionals.empty()) {
    // Convert to help names
    std::vector<std::string> positional_names(positionals.size());
    std::transform(positionals.begin(), positionals.end(), positional_names.begin(), [this](const Option *opt) {
      return make_option_usage(opt);
    });

    out << " " << detail::join(positional_names, " ");
  }

  // Add a marker if subcommands are expected or optional
  if (!app->get_subcommands(
              [](const CLI::App *subc) { return ((!subc->get_disabled()) && (!subc->get_name().empty())); })
           .empty()) {
    out << " " << (app->get_require_subcommand_min() == 0 ? "[" : "")
        << get_label(app->get_require_subcommand_max() < 2 || app->get_require_subcommand_min() > 1 ? "SUBCOMMAND"
                                                                                                    : "SUBCOMMANDS")
        << (app->get_require_subcommand_min() == 0 ? "]" : "");
  }

  out << '\n';

  return out.str();
}

CLI11_INLINE std::string Formatter::make_footer(const App *app) const {
  std::string footer = app->get_footer();
  if (footer.empty()) {
    return std::string{};
  }
  return "\n" + footer + "\n";
}

CLI11_INLINE std::string Formatter::make_help(const App *app, std::string name, AppFormatMode mode) const {

  // This immediately forwards to the make_expanded method. This is done this way so that subcommands can
  // have overridden formatters
  if (mode == AppFormatMode::Sub)
    return make_expanded(app);

  std::stringstream out;
  if ((app->get_name().empty()) && (app->get_parent() != nullptr)) {
    if (app->get_group() != "Subcommands") {
      out << app->get_group() << ':';
    }
  }

  out << make_description(app);
  out << make_usage(app, name);
  out << make_positionals(app);
  out << make_groups(app, mode);
  out << make_subcommands(app, mode);
  out << make_footer(app);

  return out.str();
}

CLI11_INLINE std::string Formatter::make_subcommands(const App *app, AppFormatMode mode) const {
  std::stringstream out;

  std::vector<const App *> subcommands = app->get_subcommands({});

  // Make a list in definition order of the groups seen
  std::vector<std::string> subcmd_groups_seen;
  for (const App *com : subcommands) {
    if (com->get_name().empty()) {
      if (!com->get_group().empty()) {
        out << make_expanded(com);
      }
      continue;
    }
    std::string group_key = com->get_group();
    if (!group_key.empty() &&
        std::find_if(subcmd_groups_seen.begin(), subcmd_groups_seen.end(), [&group_key](std::string a) {
          return detail::to_lower(a) == detail::to_lower(group_key);
        }) == subcmd_groups_seen.end())
      subcmd_groups_seen.push_back(group_key);
  }

  // For each group, filter out and print subcommands
  for (const std::string &group : subcmd_groups_seen) {
    out << "\n"
        << group << ":\n";
    std::vector<const App *> subcommands_group = app->get_subcommands(
        [&group](const App *sub_app) { return detail::to_lower(sub_app->get_group()) == detail::to_lower(group); });
    for (const App *new_com : subcommands_group) {
      if (new_com->get_name().empty())
        continue;
      if (mode != AppFormatMode::All) {
        out << make_subcommand(new_com);
      } else {
        out << new_com->help(new_com->get_name(), AppFormatMode::Sub);
        out << "\n";
      }
    }
  }

  return out.str();
}

CLI11_INLINE std::string Formatter::make_subcommand(const App *sub) const {
  std::stringstream out;
  detail::format_help(out,
                      sub->get_display_name(true) + (sub->get_required() ? " " + get_label("REQUIRED") : ""),
                      sub->get_description(),
                      column_width_);
  return out.str();
}

CLI11_INLINE std::string Formatter::make_expanded(const App *sub) const {
  std::stringstream out;
  out << sub->get_display_name(true) << "\n";

  out << make_description(sub);
  if (sub->get_name().empty() && !sub->get_aliases().empty()) {
    detail::format_aliases(out, sub->get_aliases(), column_width_ + 2);
  }
  out << make_positionals(sub);
  out << make_groups(sub, AppFormatMode::Sub);
  out << make_subcommands(sub, AppFormatMode::Sub);

  // Drop blank spaces
  std::string tmp = detail::find_and_replace(out.str(), "\n\n", "\n");
  tmp = tmp.substr(0, tmp.size() - 1); // Remove the final '\n'

  // Indent all but the first line (the name)
  return detail::find_and_replace(tmp, "\n", "\n  ") + "\n";
}

CLI11_INLINE std::string Formatter::make_option_name(const Option *opt, bool is_positional) const {
  if (is_positional)
    return opt->get_name(true, false);

  return opt->get_name(false, true);
}

CLI11_INLINE std::string Formatter::make_option_opts(const Option *opt) const {
  std::stringstream out;

  if (!opt->get_option_text().empty()) {
    out << " " << opt->get_option_text();
  } else {
    if (opt->get_type_size() != 0) {
      if (!opt->get_type_name().empty())
        out << " " << get_label(opt->get_type_name());
      if (!opt->get_default_str().empty())
        out << " [" << opt->get_default_str() << "] ";
      if (opt->get_expected_max() == detail::expected_max_vector_size)
        out << " ...";
      else if (opt->get_expected_min() > 1)
        out << " x " << opt->get_expected();

      if (opt->get_required())
        out << " " << get_label("REQUIRED");
    }
    if (!opt->get_envname().empty())
      out << " (" << get_label("Env") << ":" << opt->get_envname() << ")";
    if (!opt->get_needs().empty()) {
      out << " " << get_label("Needs") << ":";
      for (const Option *op : opt->get_needs())
        out << " " << op->get_name();
    }
    if (!opt->get_excludes().empty()) {
      out << " " << get_label("Excludes") << ":";
      for (const Option *op : opt->get_excludes())
        out << " " << op->get_name();
    }
  }
  return out.str();
}

CLI11_INLINE std::string Formatter::make_option_desc(const Option *opt) const { return opt->get_description(); }

CLI11_INLINE std::string Formatter::make_option_usage(const Option *opt) const {
  // Note that these are positionals usages
  std::stringstream out;
  out << make_option_name(opt, true);
  if (opt->get_expected_max() >= detail::expected_max_vector_size)
    out << "...";
  else if (opt->get_expected_max() > 1)
    out << "(" << opt->get_expected() << "x)";

  return opt->get_required() ? out.str() : "[" + out.str() + "]";
}

} // namespace CLI
#pragma once

#include <arpa/inet.h>
#include <stdint.h>

namespace audio {

struct __attribute__((packed)) RTPHeader {
private:
  uint8_t m_version_and_p_xx_cc;
  uint8_t m_m_and_payload_type;
  uint16_t m_seq_number;
  uint32_t m_timestamp;
  uint32_t m_ssrc_id;

  // int32_t[] csrc ids
  // int16_t profile_specific_header_id | ext header

  static constexpr uint8_t WORD0_VERSION_SHIFT = 6;
  static constexpr uint8_t WORD0_PADDING_SHIFT = 5;
  static constexpr uint8_t WORD0_EXTENSION_SHIFT = 4;
  static constexpr uint8_t WORD0_CC_SHIFT = 0;

public:
  RTPHeader(uint8_t num_cc, bool have_padding_at_end_of_pkt,
            bool is_special_packet, bool have_extension, uint8_t payload_type,
            uint16_t seq_number, uint32_t timestamp, uint32_t ssrc_id) {
    const int version = 2;

    m_version_and_p_xx_cc = (version << WORD0_VERSION_SHIFT) |
                            (have_padding_at_end_of_pkt << WORD0_PADDING_SHIFT) |
                            (have_extension << WORD0_EXTENSION_SHIFT) |
                            (num_cc << WORD0_CC_SHIFT);

    m_m_and_payload_type = (is_special_packet << 7) | (payload_type << 0);
    m_seq_number = ntohs(seq_number);
    m_timestamp = ntohl(timestamp);
    m_ssrc_id = ntohl(ssrc_id);
  }

  uint32_t get_timestamp() const { return htonl(m_timestamp); }
  uint32_t get_ssrc_id() const { return htonl(m_ssrc_id); }

  uint16_t get_seq_number() const { return htons(m_seq_number); }

  uint8_t get_version() const { return m_version_and_p_xx_cc >> WORD0_VERSION_SHIFT; }
  bool have_padding_at_end() const { return (m_version_and_p_xx_cc >> WORD0_PADDING_SHIFT) & 1; }
  bool have_extension() const { return (m_version_and_p_xx_cc >> WORD0_EXTENSION_SHIFT) & 1; }
  uint8_t get_num_cc() const { return m_version_and_p_xx_cc & 0x111; }
  uint8_t get_payload_type() const { return m_m_and_payload_type & ~(1 << 7); }
};

} // namespace audio

#include <stdint.h>

namespace SAP {
enum class AddressType : uint8_t {
  IPv4,
  IPv6
};

enum class MessageType : uint8_t {
  ANNOUNCE,
  REVOKE
};

enum class EncryptionType : uint8_t {
  ENCRYPTED,
  NONE
};

enum class CompressedType : uint8_t {
  COMPRESSED,
  NONE
};

struct __attribute__((packed)) SAPHeader {
private:
  uint8_t m_flags;
  uint8_t m_auth_len;
  uint16_t m_msg_id_hash;

public:
  static constexpr uint8_t SHIFT_VERSION_TYPE = 5;
  static constexpr uint8_t SHIFT_MESSAGE_TYPE = 3;
  static constexpr uint8_t SHIFT_ANNOUNCE_TYPE = 2;
  static constexpr uint8_t SHIFT_ENCRYPTION_TYPE = 1;
  static constexpr uint8_t SHIFT_COMPRESSED_TYPE = 0;

  SAPHeader(int auth_len, int id_hash) {
    int version = 1;
    m_flags = version << SHIFT_VERSION_TYPE;
    m_auth_len = auth_len;
    m_msg_id_hash = htons(id_hash);
  }

  uint8_t get_auth_len() const { return m_auth_len; }

  uint8_t get_version() const {
    return (m_flags >> SHIFT_VERSION_TYPE) & 0b111;
  }

  AddressType get_address_type() const {
    return m_flags & (1 << SHIFT_MESSAGE_TYPE) ? AddressType::IPv6 : AddressType::IPv4;
  }

  MessageType get_message_type() const {
    return m_flags & (1 << SHIFT_ANNOUNCE_TYPE) ? MessageType::REVOKE : MessageType::ANNOUNCE;
  }

  EncryptionType get_encryption_type() const {
    return m_flags & (1 << SHIFT_ENCRYPTION_TYPE) ? EncryptionType::ENCRYPTED : EncryptionType::NONE;
  }

  CompressedType get_compressed_type() const {
    return m_flags & (1 << SHIFT_COMPRESSED_TYPE) ? CompressedType::COMPRESSED : CompressedType::NONE;
  }
};

} // namespace SAP

#include <memory>

#include <urtsched/IService.hpp>

#include <domainmodel/Flow.hpp>

namespace model {
class Node;
}

namespace audio {
class IRTP_Service : public service::IService {
public:
  virtual void add_receive_flow(const std::shared_ptr<model::Flow> &flow) = 0;
  virtual void remove_receive_flow(const std::shared_ptr<model::Flow> &flow) = 0;
  virtual bool already_have_flow(const std::shared_ptr<model::Flow> &flow) = 0;
};

} // namespace audio

#include <optional>

#include <iuring/NetworkAdapter.hpp>

#include <urtsched/Service.hpp>

#include <domainmodel/Flow.hpp>

#include <Configuration.hpp>
#include <audio/AudioFormat.hpp>
#include <audio/AudioProfile.hpp>
#include <audio/IAudioService.hpp>
#include <ptp/PtpService.hpp>
#include <statistics/Histogram.hpp>

#include "IRTP_Service.hpp"
#include <domainmodel/ChannelMapper.hpp>

#include <audio/InterleavedAudioFrame.hpp>

#include <srtp.h>

namespace model {
class Node;
}

namespace audio {
/** this service sends/receives RTP audio packets */
class RTP_Service : virtual public IRTP_Service, virtual public service::Service {
public:
  RTP_Service(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
              const std::shared_ptr<iuring::IOUringInterface> &network,
              logging::ILogger &logger,
              const std::shared_ptr<ptp::PtpService> &ptp_service,
              const settings::Configuration &config,
              iuring::NetworkAdapter &adapter,
              iuring::ISocketFactory &socket_factory,
              model::ChannelMapper &mapper,
              std::shared_ptr<model::Node> model)
      : service::Service(rt_kernel, logger), m_socket_factory(socket_factory), m_ptp_service(ptp_service), m_config(config), m_adapter(adapter), m_network(network), m_mapper(mapper), m_model(model) {
  }

  std::shared_ptr<iuring::IOUringInterface> &get_io() {
    return m_network;
  }

  std::string get_service_status_as_json() const override;

  [[nodiscard]] error::Error init() override;
  error::Error finish() override {
    // Clean up SRTP contexts
    if (m_srtp_send_ctx) {
      srtp_dealloc(m_srtp_send_ctx);
      m_srtp_send_ctx = nullptr;
    }
    if (m_srtp_recv_ctx) {
      srtp_dealloc(m_srtp_recv_ctx);
      m_srtp_recv_ctx = nullptr;
    }
    if (m_config.enable_srtp_encryption) {
      srtp_shutdown();
    }
    return error::Error::OK;
  }

  void add_receive_flow(const std::shared_ptr<model::Flow> &flow) override;
  void remove_receive_flow(const std::shared_ptr<model::Flow> &flow) override;
  bool already_have_flow(const std::shared_ptr<model::Flow> &flow) override;

  std::string sent_packets_histogram_json() const {
    return m_histogram.sent_to_json_string();
  }

  std::string recv_packets_histogram_json() const {
    return m_histogram.recv_to_json_string();
  }

  uint64_t num_rtp_packets_sent() const {
    return m_histogram.num_rtp_packets_sent();
  }

  uint64_t num_rtp_packets_received() const {
    return m_histogram.num_packets_received();
  }

private:
  bool m_init = false;
  iuring::ISocketFactory &m_socket_factory;
  int m_num_in_flight = 0;

  utils::Histogram m_histogram;

  std::shared_ptr<ptp::PtpService> m_ptp_service;
  uint32_t m_rtp_seq_number = 1234;
  std::shared_ptr<iuring::ISocket> m_rtp_socket;
  std::shared_ptr<realtime::PeriodicTask> m_rtp_task;
  const settings::Configuration &m_config;
  iuring::NetworkAdapter &m_adapter;
  std::shared_ptr<iuring::IOUringInterface> m_network;
  model::ChannelMapper &m_mapper;
  std::shared_ptr<model::Node> m_model;

  // Pre-allocated interleave buffer – avoids stack page-faults in the RT task
  InterleavedAudioFrame m_send_frame{AudioEncodingID::L24_2CH, "rtp-send"};

  // SRTP encryption contexts
  srtp_t m_srtp_send_ctx = nullptr;
  srtp_t m_srtp_recv_ctx = nullptr;

  void send_rtp_audio_packets();

  void handle_incoming_rtp_packet(
      const iuring::ReceivedMessage &data, const AudioFormat &format);

  void try_send_audio_frame(InterleavedAudioFrame &chunk);

  iuring::NetworkAdapter &get_adapter() {
    return m_adapter;
  }
};

} // namespace audio

#include <slogger/ILogger.hpp>
#include <slogger/TimeUtils.hpp>

#include <urtsched/IService.hpp>
#include <urtsched/ServiceBus.hpp>

#include <iuring/IOUringInterface.hpp>

#include <audio/IAudioService.hpp>
#include <ptp/PtpService.hpp>

#include <aoip/IRTP_Service.hpp>
#include <aoip/SAP_Service.hpp>
#include <mdns/MDNS_Service.hpp>

namespace audio {
class RavennaService : public service::IService {
public:
  RavennaService(service::ServiceBus &bus,
                 const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                 const std::shared_ptr<iuring::IOUringInterface> &network,
                 logging::ILogger &logger,
                 const std::shared_ptr<ptp::IPtpService> &ptp_service,
                 const std::shared_ptr<mdns::MDNS_Service> &mdns_server,
                 const settings::Configuration &config,
                 iuring::NetworkAdapter &adapter,
                 const std::shared_ptr<IRTP_Service> &rtp_service,
                 iuring::ISocketFactory &socket_factory,
                 const std::shared_ptr<model::Node> &node);

  error::Error init() override;
  error::Error finish() override { return error::Error::OK; }

  uint32_t num_sap_packets_sent() const {
    return m_sap_service->num_sap_packets_sent();
  }

  uint32_t num_sap_packets_received() const {
    return m_sap_service->num_sap_packets_received();
  }

private:
  service::ServiceBus &m_bus;
  std::shared_ptr<IRTP_Service> m_rtp_service;
  std::shared_ptr<SAP_Service> m_sap_service;
  std::shared_ptr<mdns::MDNS_Service> m_mdns_server;
  const settings::Configuration &m_config;
};
} // namespace audio

#include <urtsched/Service.hpp>

#include <iuring/ISocketFactory.hpp>

#include <audio/AudioProfile.hpp>
#include <audio/IAudioService.hpp>

#include <ptp/PtpService.hpp>

#include <Configuration.hpp>

namespace model {
class Node;
}

namespace audio {
class RTP_Service;

class SAP_Service : public service::Service {
public:
  SAP_Service(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
              const std::shared_ptr<iuring::IOUringInterface> &network,
              logging::ILogger &logger,
              const std::shared_ptr<ptp::IPtpService> &ptp_service,
              const std::shared_ptr<IRTP_Service> &rtp_manager,
              const settings::Configuration &config,
              iuring::NetworkAdapter &adapter,
              iuring::ISocketFactory &socket_factory,
              const std::shared_ptr<model::Node> &model)
      : service::Service(rt_kernel, logger), m_socket_factory(socket_factory), m_ptp_service(ptp_service), m_rtp_manager(rtp_manager), m_config(config), m_adapter(adapter), m_network(network), m_model(model) {
  }

  void send_sap_announce_packet();

  std::string get_service_status_as_json() const override;

  [[nodiscard]] error::Error init();
  [[nodiscard]] error::Error finish() override { return error::Error::OK; }

  void handle_incoming_sap_event(const iuring::ReceivedMessage &data);

  uint32_t num_sap_packets_sent() const {
    return m_num_sap_packets_sent;
  }

  uint32_t num_sap_packets_received() const {
    return m_num_sap_packets_received;
  }

private:
  static iuring::IPAddress SAP_PRIMARY_MCAST_IPADDR;

  iuring::ISocketFactory &m_socket_factory;

  uint32_t m_num_sap_packets_sent = 0;
  uint32_t m_num_sap_packets_received = 0;

  std::shared_ptr<ptp::IPtpService> m_ptp_service;
  std::shared_ptr<IRTP_Service> m_rtp_manager;
  std::shared_ptr<realtime::PeriodicTask> m_sap_announce_task;

  std::shared_ptr<iuring::ISocket> m_sap_event_socket;

  const settings::Configuration &m_config;
  iuring::NetworkAdapter &m_adapter;
  std::shared_ptr<iuring::IOUringInterface> m_network;

  std::shared_ptr<model::Node> m_model;

  iuring::NetworkAdapter &get_adapter() {
    return m_adapter;
  }

  const std::shared_ptr<iuring::IOUringInterface> get_io() {
    return m_network;
  }

  void handle_announce(const std::string &sdp);
};

/**
 * @brief Create an SDP description for the given parameters.
 *
 * @param rtp_multicast_address The RTP multicast address (IP), IRTP_Service::RTP_AUDIO_MCAST_IPADDR by default
 *
 */
std::string create_sdp_description(const iuring::IPAddress &rtp_multicast_address,
                                   bool use_ipv4,
                                   const iuring::IPAddress &my_address,
                                   const ptp::v2::clock_identity_t &my_clk_id,
                                   const ptp::v2::clock_identity_t &master_clk_id,
                                   const settings::Configuration &config);

} // namespace audio
#pragma once

#include <expected>
#include <string>
#include <vector>

#include <slogger/Error.hpp>
#include <slogger/ILogger.hpp>

#include <domainmodel/Flow.hpp>

namespace aoip {
class SDP_Data {
public:
  SDP_Data(logging::ILogger &logger,
           const std::shared_ptr<model::Source> &parent_source)
      : m_logger(logger), m_parent_source(parent_source) {
  }

  std::expected<std::shared_ptr<model::Flow>, error::Error> parse(
      const std::string &sdp_description);

private:
  logging::ILogger &m_logger;
  std::shared_ptr<model::Source> m_parent_source;

  bool parse_line(model::Flow &flow, const std::string &trimmed, bool &saw_m);
  bool sdp_parse_version(model::Flow &flow, const std::string &value);
  bool sdp_parse_owner(model::Flow &flow, const std::string &value);
  bool sdp_parse_session_info(model::Flow &flow, const std::string &value);
  bool sdp_parse_channel_info(model::Flow &flow, const std::string &value);
  bool sdp_parse_channel_address_info(
      model::Flow &flow, const std::string &value);
  bool sdp_parse_session_valid_time(
      model::Flow &flow, const std::string &value);
  bool sdp_parse_session_attribute(
      model::Flow &flow, const std::string &value);
  bool sdp_parse_stream_attribute(
      model::Flow &flow, const std::string &value);
  bool sdp_parse_audio_properties(
      model::Flow &flow, const std::string &value);

  logging::ILogger &get_logger() {
    return m_logger;
  }
};
} // namespace aoip

#include <chrono>

#include <audio/AudioFormat.hpp>

namespace audio {
struct AudioProfile {
  // every N millis send a SAP announcement
  std::chrono::milliseconds SAP_ANNOUNCE_INTERVAL;

  // what encoding to use
  AudioFormat m_audio_format_input;  // for recording from HW
  AudioFormat m_audio_format_output; // for playback to HW

  // SAP protocol ID
  int rtp_payload_type; // >= 97 used by SDP media type and in RTP packet

  bool is_audio_server;
  bool is_audio_client;

  void set_input_format(uint32_t num_channels, uint32_t num_bits,
                        uint32_t sample_rate);
  void set_output_format(uint32_t num_channels, uint32_t num_bits,
                         uint32_t sample_rate);
};

extern const AudioProfile default_ravenna_profile;

} // namespace audio

#include <cassert>
#include <chrono>
#include <cstdint>
#include <stdlib.h>

#include <iuring/NetworkProtocols.hpp>

namespace network {

struct NetworkingProfile {
  uint8_t domain;
  std::chrono::milliseconds PTP_AES67_PING_INTERVAL_V1;
  std::chrono::milliseconds PTP_AES67_PING_INTERVAL_V2;

  std::chrono::milliseconds PTP_AES67_SYNC_INTERVAL_V1;
  std::chrono::milliseconds PTP_AES67_SYNC_INTERVAL_V2;

  std::chrono::milliseconds PTP_AES67_ANNOUNCE_INTERVAL;
  std::chrono::milliseconds PTP_AES67_ANNOUNCE_TIMEOUT;

  std::chrono::milliseconds PTP_AES67_DELAY_REQUEST_INTERVAL;

  iuring::dscp_t DSCP_for_events;
  iuring::dscp_t DSCP_for_general;
  iuring::dscp_t DSCP_for_RTP;
  iuring::timetolive_t rtp_ttl;
  iuring::timetolive_t ptp_ttl;

  // PTP v1 prefs
  int m_v1_stratum = 210;
  bool m_v1_boundary_clock = true;
  bool m_v1_prefered_clock = false;
  std::chrono::milliseconds listen_timeout_v1 = std::chrono::milliseconds(5000);

  // PTP v2 prefs
  int m_v2_priority1 = 249;
  int m_v2_priority2 = 120;
  int m_v2_clock_class = 248;
  std::chrono::milliseconds listen_timeout_v2 = std::chrono::milliseconds(5000);

  int get_v1_stratum() const { return m_v1_stratum; }

  int get_ptpv1_announce_interval_value() const {
    return millis_to_pow2(PTP_AES67_ANNOUNCE_INTERVAL);
  }

  int get_ptpv1_sync_interval_value() const {
    return millis_to_pow2(PTP_AES67_SYNC_INTERVAL_V1);
  }

  static int millis_to_pow2(const std::chrono::milliseconds &v) {
    if (v == std::chrono::milliseconds(1000 * 16))
      return 4;
    if (v == std::chrono::milliseconds(1000 * 8))
      return 3;
    if (v == std::chrono::milliseconds(1000 * 4))
      return 2;
    if (v == std::chrono::milliseconds(1000 * 2))
      return 1;
    if (v == std::chrono::milliseconds(1000))
      return 0;
    if (v == std::chrono::milliseconds(1000 / 2))
      return -1;
    if (v == std::chrono::milliseconds(1000 / 4))
      return -2;
    if (v == std::chrono::milliseconds(1000 / 8))
      return -3;
    if (v == std::chrono::milliseconds(1000 / 16))
      return -4;
    assert(false);
    return 0;
  }

  int get_v2_sync_log_interval() const {
    return millis_to_pow2(PTP_AES67_SYNC_INTERVAL_V2);
  }
};

extern const NetworkingProfile AES76_profile;
extern const NetworkingProfile SMPTE_ST2059_2_profile;

} // namespace network
#pragma once

#include <memory>

#include <alsa/asoundlib.h>

#include <domainmodel/ChannelMapper.hpp>

#include <audio/IAlsaManager.hpp>
#include <audio/IAudioService.hpp>
#include <audio/InterleavedAudioFrame.hpp>
#include <urtsched/RealtimeKernel.hpp>

#include <Configuration.hpp>

namespace audio {

class AlsaAudioService : public IAudioService {
public:
  AlsaAudioService(logging::ILogger &logger,
                   bool is_input,
                   bool is_output,
                   const std::string &device_name,
                   const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                   settings::Configuration &config, model::ChannelMapper &mapper,
                   const std::shared_ptr<model::Device> &device,
                   const std::string &description,
                   const std::shared_ptr<ptp::IPtpService> &ptp_service)
      : IAudioService(logger, config.audio_profile, rt_kernel, device, ptp_service), m_is_input(is_input), m_is_output(is_output), m_device_name(device_name), m_config(config), m_mapper(mapper), m_description(description) {
  }

  std::string get_name() const override { return m_device_name; }
  std::string get_description() const override { return m_description; }

  std::vector<std::string> get_hw_input_channel_names() const override;
  std::vector<std::string> get_hw_output_channel_names() const override;

  channel_index_t get_num_hw_input_channels();
  channel_index_t get_num_hw_output_channels();

  error::Error init() final;
  error::Error finish() final;

private:
  bool m_initialized = false;

  const bool m_is_input;
  const bool m_is_output;

  const std::string m_device_name;

  snd_pcm_t *pcm_playback_handle = nullptr;
  snd_pcm_t *pcm_recording_handle = nullptr;

  std::shared_ptr<realtime::IdleTask> m_playback_task;
  std::shared_ptr<realtime::IdleTask> m_recording_task;

  snd_pcm_hw_params_t *recording_hw_params = nullptr;
  snd_pcm_hw_params_t *playback_hw_params = nullptr;

  uint32_t m_num_recording_channels = 0;
  uint32_t m_num_playback_channels = 0;

  snd_pcm_uframes_t recording_frames{};
  snd_pcm_uframes_t playback_frames{};

  settings::Configuration &m_config;

  model::ChannelMapper &m_mapper;
  std::string m_description;

  error::Error init_playback();
  error::Error init_recording();

  void poll_playout_possible();
  void poll_recording_available();

  void playback(const AudioFrame &chunk);
  bool record(AudioFrame &chunk);

  // Pre-allocated interleave buffers — avoids 61 KB stack frames in the RT path.
  InterleavedAudioFrame m_playback_frame{AudioEncodingID::L24_2CH, "alsa-playback"};
  InterleavedAudioFrame m_record_frame{AudioEncodingID::L24_2CH, "alsa-recording"};
};

} // namespace audio

#include <audio/IAlsaManager.hpp>

namespace audio {

class AlsaManager : public IAlsaManager {
public:
  std::vector<std::shared_ptr<AlsaAudioService>> get_devices(
      logging::ILogger &logger,
      const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
      settings::Configuration &config, model::ChannelMapper &mapper,
      const std::shared_ptr<ptp::IPtpService> &ptp_service) override;
};

} // namespace audio
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace audio {
using channel_index_t = uint16_t;

enum class AudioEncodingID {
  UNKNOWN,

#define CH_ENC_MACRO(C) \
  L8_##C##CH,           \
      L16_##C##CH,      \
      L24_##C##CH,      \
      L32_##C##CH,

  CH_ENC_MACRO(1)
      CH_ENC_MACRO(2)
          CH_ENC_MACRO(4)
              CH_ENC_MACRO(8)
                  CH_ENC_MACRO(16)
                      CH_ENC_MACRO(32)
                          CH_ENC_MACRO(64)
                              CH_ENC_MACRO(128)

};

class AudioFormat {
public:
  /**
   * bitwidth: 8/16/24/32
   */
  AudioFormat(AudioEncodingID encoding = AudioEncodingID::L24_2CH, uint32_t sample_rate = 48000)
      : m_encoding(encoding), m_sample_rate(sample_rate) {
  }

  bool is_for_single_channel() const {
    switch (m_encoding) {
    case AudioEncodingID::UNKNOWN:
      fprintf(stderr, "called is_for_single_channel for unknown");
      abort();
    case AudioEncodingID::L8_1CH:
    case AudioEncodingID::L16_1CH:
    case AudioEncodingID::L24_1CH:
    case AudioEncodingID::L32_1CH:
      return true;

    default:
      return false;
    }
    abort();
  }

  void set_sample_rate(uint32_t sample_rate) {
    m_sample_rate = sample_rate;
  }

  void set_num_channels(uint32_t num_channels) {
    switch (m_encoding) {
    case AudioEncodingID::UNKNOWN:
      fprintf(stderr, "called set_num_channels for unknown");
      abort();

#define CHANGE_DISPATCH_CH(BITS, X)                                        \
  switch (num_channels) {                                                  \
  case 1:                                                                  \
    m_encoding = AudioEncodingID::BITS##_1CH;                              \
    break;                                                                 \
  case 2:                                                                  \
    m_encoding = AudioEncodingID::BITS##_2CH;                              \
    break;                                                                 \
  case 4:                                                                  \
    m_encoding = AudioEncodingID::BITS##_4CH;                              \
    break;                                                                 \
  case 8:                                                                  \
    m_encoding = AudioEncodingID::BITS##_8CH;                              \
    break;                                                                 \
  case 16:                                                                 \
    m_encoding = AudioEncodingID::BITS##_16CH;                             \
    break;                                                                 \
  case 32:                                                                 \
    m_encoding = AudioEncodingID::BITS##_32CH;                             \
    break;                                                                 \
  case 64:                                                                 \
    m_encoding = AudioEncodingID::BITS##_64CH;                             \
    break;                                                                 \
  case 128:                                                                \
    m_encoding = AudioEncodingID::BITS##_128CH;                            \
    break;                                                                 \
  default:                                                                 \
    fprintf(stderr, "unsupported number of channels: %d\n", num_channels); \
    abort();                                                               \
    break;                                                                 \
  }                                                                        \
  break;

#define DISPATCH_CH(X)               \
  case AudioEncodingID::L8_##X##CH:  \
    CHANGE_DISPATCH_CH(L8, X)        \
  case AudioEncodingID::L16_##X##CH: \
    CHANGE_DISPATCH_CH(L16, X)       \
  case AudioEncodingID::L24_##X##CH: \
    CHANGE_DISPATCH_CH(L24, X)       \
  case AudioEncodingID::L32_##X##CH: \
    CHANGE_DISPATCH_CH(L32, X)

      DISPATCH_CH(1);
      DISPATCH_CH(2);
      DISPATCH_CH(4);
      DISPATCH_CH(8);
      DISPATCH_CH(16);
      DISPATCH_CH(32);
      DISPATCH_CH(64);
      DISPATCH_CH(128);

#undef CHANGE_DISPATCH_CH
#undef DISPATCH_CH
    }
  }

  AudioEncodingID get_encoding() const { return m_encoding; }
  uint32_t get_sample_rate() const { return m_sample_rate; }

  size_t num_samples_for_1_millisecond_for_single_channel() const {
    return m_sample_rate / 1000;
  }

  size_t get_size_for_1_millisecond() const {
    return get_size_for_1_second() / 1000;
  }

  size_t get_bytes_per_sample() const {
    switch (m_encoding) {
    case AudioEncodingID::UNKNOWN:
      fprintf(stderr, "called get_bytes_per_sample for unknown");
      abort();

#define DISPATCH_CH(X)               \
  case AudioEncodingID::L8_##X##CH:  \
    return 1;                        \
  case AudioEncodingID::L16_##X##CH: \
    return 2;                        \
  case AudioEncodingID::L24_##X##CH: \
    return 3;                        \
  case AudioEncodingID::L32_##X##CH: \
    return 4;

      DISPATCH_CH(1);
      DISPATCH_CH(2);
      DISPATCH_CH(4);
      DISPATCH_CH(8);
      DISPATCH_CH(16);
      DISPATCH_CH(32);
      DISPATCH_CH(64);
      DISPATCH_CH(128);

#undef DISPATCH_CH
    }
    abort();
  }

  channel_index_t get_num_channels() const {
    switch (m_encoding) {
    case AudioEncodingID::UNKNOWN:
      fprintf(stderr, "called get_num_channels for unknown");
      abort();

#define DISPATCH_CH(X)               \
  case AudioEncodingID::L8_##X##CH:  \
    return X;                        \
  case AudioEncodingID::L16_##X##CH: \
    return X;                        \
  case AudioEncodingID::L24_##X##CH: \
    return X;                        \
  case AudioEncodingID::L32_##X##CH: \
    return X;

      DISPATCH_CH(1);
      DISPATCH_CH(2);
      DISPATCH_CH(4);
      DISPATCH_CH(8);
      DISPATCH_CH(16);
      DISPATCH_CH(32);
      DISPATCH_CH(64);
      DISPATCH_CH(128);

#undef DISPATCH_CH
    }
    abort();
  }

  uint32_t get_bitwidth() const {
    return get_bytes_per_sample() * 8;
  }

  size_t get_size_for_1_second() const {
    return get_sample_rate() * get_bytes_per_sample() * get_num_channels();
  }

private:
  AudioEncodingID m_encoding;
  uint32_t m_sample_rate;
};
} // namespace audio

#include "SampleChunkSingleChannel.hpp"

namespace audio {

/** samples from multiple channels at a given point in time */
class AudioFrame {
public:
  AudioFrame() {}

  AudioFrame(const channel_index_t num_channels)
      : m_num_channels(num_channels) {
  }

  void set_num_channels(const channel_index_t num_channels) {
    assert(num_channels <= MAX_NUMBER_OF_CHANNELS);
    m_num_channels = num_channels;
  }

  void mark_finished() {
    for (channel_index_t i = 0; i < m_num_channels; i++) {
      m_chunks[i].mark_finished();
    }
  }

  SampleChunkSingleChannel &get_channel(channel_index_t channel) {
    assert(channel < m_num_channels);
    return m_chunks[channel];
  }

  std::array<SampleChunkSingleChannel, MAX_NUMBER_OF_CHANNELS> &get_all_channels() {
    return m_chunks;
  }

  const SampleChunkSingleChannel &get_channel(channel_index_t channel) const {
    assert(channel < m_num_channels);
    return m_chunks[channel];
  }

  void consumer_reset() {
    for (auto &c : m_chunks) {
      c.consumer_reset();
    }
  }

  rtp_clock_timestamp_t get_logical_sample_clock_for_first_sample() const {
    return m_chunks[0].get_logical_sample_clock_for_first_sample();
  }

  channel_index_t get_num_channels() const {
    return m_num_channels;
  }

  void import_channels(const uint8_t *data, size_t len,
                       const AudioFormat &format,
                       const std::chrono::nanoseconds &ptp_time, logging::ILogger &logger);

private:
  channel_index_t m_num_channels = 0;
  std::array<SampleChunkSingleChannel, MAX_NUMBER_OF_CHANNELS> m_chunks;
};

} // namespace audio

#include <cstddef>
#include <cstdint>

namespace audio {
static constexpr size_t MAX_SAMPLE_RATE = 192000; // 192 kHz

static constexpr size_t MAX_TIME_PERIOD_MS = 10; // 10 milliseconds

static constexpr size_t MAX_NUMBER_OF_SAMPLES_PER_TIME_PERIOD =
    (MAX_SAMPLE_RATE * MAX_TIME_PERIOD_MS) / 1000;

static constexpr size_t MAX_NUMBER_OF_CHANNELS = 8;

static constexpr size_t MAX_BYTES_PER_SAMPLE = 4;

static constexpr size_t MAX_INTERLEAVED_AUDIO_BUFFER_SIZE =
    MAX_NUMBER_OF_SAMPLES_PER_TIME_PERIOD * MAX_NUMBER_OF_CHANNELS *
    MAX_BYTES_PER_SAMPLE;
} // namespace audio
#pragma once

#include <memory>
#include <set>
#include <string>

#include <slogger/Error.hpp>
#include <slogger/ILogger.hpp>

#include <urtsched/IService.hpp>
#include <urtsched/RealtimeKernel.hpp>

#include <domainmodel/Node.hpp>

namespace audio {

/** Copies data from one IAudioDevice's capture to another's playback,
 * based on the settings of the ChannelMapper in the model.
 */
class AudioPipeline : public service::IService {
public:
  AudioPipeline(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                logging::ILogger &logger,
                const std::shared_ptr<model::Node> &model)
      : m_rt_kernel(rt_kernel), m_logger(logger), m_model(model) {
  }

  logging::ILogger &get_logger() const {
    return m_logger;
  }

  error::Error init() override;

  error::Error finish() override {
    return error::Error::OK;
  }

private:
  std::shared_ptr<realtime::RealtimeKernel> m_rt_kernel;
  logging::ILogger &m_logger;
  std::shared_ptr<model::Node> m_model;
  std::shared_ptr<realtime::PeriodicTask> m_task;

  void copy_data_from_captures_to_playbacks();
  void apply_mappings(std::set<std::pair<std::string, channel_index_t>> &mapped_channels);
  void apply_single_mapping(const model::Mapping &mapping,
                            std::set<std::pair<std::string, channel_index_t>> &mapped_channels);
  void silence_unmapped_channels(const std::set<std::pair<std::string, channel_index_t>> &mapped_channels);
  size_t find_expected_samples(const AudioFrame &playback_frame,
                               const std::set<std::pair<std::string, channel_index_t>> &mapped_channels,
                               const std::string &output_id) const;
  void silence_output_channels(AudioFrame &playback_frame,
                               const std::set<std::pair<std::string, channel_index_t>> &mapped_channels,
                               const std::string &output_id, size_t expected_samples);
};

} // namespace audio

#include <array>
#include <cassert>
#include <cstdint>

#include <audio/AudioFormat.hpp>
#include <audio/AudioFrame.hpp>

namespace audio {

/** An Audio Stream consists of multiple audio channels, each with its own
 * SampleChunk array */
struct AudioStream {
public:
  using RecordFrame = AudioFrame;
  using PlaybackFrame = AudioFrame;

  AudioStream() {}

  /** let the consumer swap buffers: this is the audio hardware that does the
   * flipping when its finished with its current buffer. If the producer is
   * still writing to the other buffer, then there will be audio dropouts
   * (underruns). If the producer is faster than the consumer, then there will
   * be overruns.
   */
  void consumer_swap_frames() {
    // Our consumed buffers are now free to be produced into:
    playback_consumer_frame().consumer_reset();
    capture_consumer_frame().consumer_reset();
    m_phase = !m_phase;
  }

  void set_num_capture_channels(channel_index_t num_channels) {
    capture_producer_frame().set_num_channels(num_channels);
    capture_consumer_frame().set_num_channels(num_channels);
  }

  void set_num_playback_channels(channel_index_t num_channels) {
    playback_producer_frame().set_num_channels(num_channels);
    playback_consumer_frame().set_num_channels(num_channels);
  }

  // recording:
  RecordFrame &capture_producer_frame() {
    return m_record_frames[m_phase];
  }

  RecordFrame &capture_consumer_frame() {
    return m_record_frames[!m_phase];
  }

  const RecordFrame &capture_producer_frame() const {
    return m_record_frames[m_phase];
  }

  const RecordFrame &capture_consumer_frame() const {
    return m_record_frames[!m_phase];
  }

  // playback:
  PlaybackFrame &playback_producer_frame() {
    return m_playback_frames[m_phase];
  }

  PlaybackFrame &playback_consumer_frame() {
    return m_playback_frames[!m_phase];
  }

  const PlaybackFrame &playback_producer_frame() const {
    return m_playback_frames[m_phase];
  }

  const PlaybackFrame &playback_consumer_frame() const {
    return m_playback_frames[!m_phase];
  }

private:
  uint8_t m_phase = 0;
  std::array<RecordFrame, 2> m_record_frames;
  std::array<PlaybackFrame, 2> m_playback_frames;
};

} // namespace audio

#include <memory>

#include <urtsched/RealtimeKernel.hpp>

#include <audio/IAudioService.hpp>

namespace audio {
class DummyAudioService : public IAudioService {
public:
  DummyAudioService(logging::ILogger &logger, AudioProfile &profile,
                    const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                    const std::shared_ptr<model::Device> &device,
                    const std::shared_ptr<ptp::IPtpService> &ptp_service)
      : IAudioService(logger, profile, rt_kernel, device, ptp_service) {
  }

  std::string get_name() const override { return "dummy"; }
  std::string get_description() const { return "dummy-descr"; }

  std::vector<std::string> get_hw_input_channel_names() const { return {"dummy"}; }
  std::vector<std::string> get_hw_output_channel_names() const { return {"dummy"}; }

protected:
  void playback(SampleChunkSingleChannel *chunk);
};

} // namespace audio

#include <memory>
#include <vector>

#include <Configuration.hpp>
#include <domainmodel/ChannelMapper.hpp>
#include <ptp/IPtpService.hpp>
#include <slogger/ILogger.hpp>
#include <urtsched/RealtimeKernel.hpp>

namespace audio {
class AlsaAudioService;

class IAlsaManager {
public:
  virtual ~IAlsaManager() = default;

  virtual std::vector<std::shared_ptr<AlsaAudioService>> get_devices(
      logging::ILogger &logger,
      const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
      settings::Configuration &config, model::ChannelMapper &mapper,
      const std::shared_ptr<ptp::IPtpService> &ptp_service) = 0;
};

} // namespace audio
#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>

#include <slogger/Error.hpp>
#include <slogger/ILogger.hpp>

#include <urtsched/IService.hpp>
#include <urtsched/RealtimeKernel.hpp>

#include <audio/AudioFormat.hpp>
#include <audio/AudioProfile.hpp>
#include <audio/AudioStream.hpp>

namespace model {
class Device;
}

namespace audio {
/** can start a new thread to do the audio stuff in the background
 */
class IAudioService : public service::IService {
public:
  IAudioService(logging::ILogger &logger, AudioProfile &profile,
                const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                const std::shared_ptr<model::Device> &dev,
                const std::shared_ptr<ptp::IPtpService> &ptp_service)
      : m_rt_kernel(rt_kernel), m_logger(logger), m_profile(profile), m_device(dev), m_ptp_service(ptp_service) {
  }

  /** if overriden, need to call base init too.
   */
  virtual error::Error init();
  virtual error::Error finish();

  virtual ~IAudioService();

  logging::ILogger &get_logger() const {
    return m_logger;
  }

  virtual std::string get_name() const = 0;
  virtual std::string get_description() const = 0;

  virtual std::vector<std::string> get_hw_input_channel_names() const = 0;
  virtual std::vector<std::string> get_hw_output_channel_names() const = 0;

  std::shared_ptr<model::Device> get_device() const {
    return m_device;
  }

  void set_device(const std::shared_ptr<model::Device> &d) {
    m_device = d;
  }

  // user of class can remove from its recorded data here
  const std::shared_ptr<AudioStream> &get_audio_stream() const {
    return m_audio_stream;
  }

  AudioProfile &get_profile() {
    return m_profile;
  }

  const std::shared_ptr<ptp::IPtpService> &get_ptp_service() const {
    return m_ptp_service;
  }

  std::shared_ptr<realtime::RealtimeKernel> &get_rt_kernel() {
    return m_rt_kernel;
  }

private:
  std::shared_ptr<realtime::RealtimeKernel> m_rt_kernel;
  logging::ILogger &m_logger;
  AudioProfile &m_profile;
  std::shared_ptr<model::Device> m_device;
  std::shared_ptr<ptp::IPtpService> m_ptp_service;
  std::shared_ptr<AudioStream> m_audio_stream = std::make_shared<AudioStream>();
};

} // namespace audio

#include "AudioFrame.hpp"
#include "AudioLimits.hpp"

namespace audio {
class InterleavedAudioFrame {
public:
  InterleavedAudioFrame(
      AudioEncodingID sample_format, const std::string &context_info)
      : m_sample_format(sample_format), m_context_info(context_info) {
  }

  /** copy individual channels to interleaved format
   * @param frame the AudioFrame to copy samples from
   */
  InterleavedAudioFrame(const AudioFrame &frame,
                        AudioEncodingID sample_format, logging::ILogger &logger,
                        const std::string &context_info);

  /** Re-populate this frame in-place from an AudioFrame. Avoids
   * re-allocating the internal 61 KB buffer on every RT call.
   */
  void populate(const AudioFrame &frame,
                AudioEncodingID sample_format, logging::ILogger &logger);

  /** size in bytes */
  size_t get_size_in_bytes() const {
    return m_size;
  }

  const uint8_t *get_data() const {
    return m_data.data();
  }

  uint8_t *get_data() {
    return m_data.data();
  }

  rtp_clock_timestamp_t get_logical_sample_clock_for_first_sample() const {
    return m_timestamp;
  }

  /** export interleaved data to individual channels in 'frame'
   * @param frame the AudioFrame to export samples into
   * @param bytes_per_sample number of bytes per sample (e.g., 3 for 24 bit
   * audio)
   * @param format the audio format of the interleaved data
   * @param ptp_time the PTP time associated with the first sample in the data
   */
  void export_to_frame(AudioFrame &frame, size_t bytes_per_sample,
                       const AudioFormat &format, const std::chrono::nanoseconds &ptp_time,
                       logging::ILogger &logger) const;

private:
  AudioEncodingID m_sample_format;
  rtp_clock_timestamp_t m_timestamp = 0;
  channel_index_t m_num_channels = 0;
  size_t m_size = 0;
  std::string m_context_info;
  std::array<uint8_t, MAX_INTERLEAVED_AUDIO_BUFFER_SIZE> m_data;
};
} // namespace audio

#include <memory>

#include <alsa/asoundlib.h>

#include <domainmodel/ChannelMapper.hpp>

#include <audio/IAudioService.hpp>
#include <urtsched/RealtimeKernel.hpp>

#include <Configuration.hpp>

namespace audio {

/** this class presents an incoming RTP stream as an audio device */
class RTPReceiverAudioService : public IAudioService {
public:
  RTPReceiverAudioService(logging::ILogger &logger,
                          const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                          settings::Configuration &config, model::ChannelMapper &mapper,
                          const iuring::IPAddress &rtp_source_address,
                          const std::shared_ptr<model::Device> &device,
                          const std::shared_ptr<ptp::IPtpService> &ptp_service)
      : IAudioService(
            logger, config.audio_profile, rt_kernel, device, ptp_service),
        m_config(config), m_mapper(mapper), m_rtp_source_address(rtp_source_address) {
  }

  std::string get_name() const { return "rtp"; }
  std::string get_description() const { return "rtp-descr"; }

  std::vector<std::string> get_hw_output_channel_names() const;
  std::vector<std::string> get_hw_input_channel_names() const;

  void handle_received_packet(const uint8_t *data, size_t len,
                              const std::chrono::nanoseconds &recv_timestamp,
                              const AudioFormat &format);

  error::Error init() final;
  error::Error finish() final;

private:
  bool m_initialized = false;

  settings::Configuration &m_config;
  model::ChannelMapper &m_mapper;
  iuring::IPAddress m_rtp_source_address;
};

} // namespace audio

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <audio/AudioFormat.hpp>
#include <audio/AudioLimits.hpp>
#include <ptp/IPtpService.hpp>

namespace audio {

using rtp_clock_timestamp_t = uint32_t;

using SampleType = int32_t;

/** contains the audio samples for a single audio channel.
 * If we have a stereo audio stream, there are two of these,
 * one for the left, one for right channel.
 */
class SampleChunkSingleChannel {
public:
  bool try_lock(int id);
  void unlock(int id);

  std::mutex mutex;
  int m_owner = 0;

  /**
   * Reset the chunk to be reused by the producer.
   */
  void consumer_reset();
  void producer_reset();

  void fill(const SampleType value, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
      m_data[i] = value;
    }
    m_size = count;
  }

  void mark_finished() {
    m_finished = true;
  }

  bool is_finished() const {
    return m_finished;
  }

  SampleType *get_data() {
    return m_data.data();
  }

  const SampleType *get_data() const {
    return m_data.data();
  }

  size_t max_size() const {
    return m_data.size();
  }

  const SampleType *end() const {
    return m_data.data() + max_size();
  }

  const SampleType *begin() const {
    return m_data.data();
  }

  void set_format(const AudioFormat &format, size_t size,
                  const std::chrono::nanoseconds &ptp_service) {
    m_format = format;
    m_format.set_num_channels(1);
    m_size = size;
    m_logical_clock = ptp_service;
    assert(m_size < m_data.size());
    assert(format.is_for_single_channel());
  }

  SampleType get_sample(size_t index) const {
    assert(index < max_size());
    return m_data[index];
  }

  void set_sample(size_t index, SampleType sample) {
    assert(index < max_size());
    m_data[index] = sample;
  }

  rtp_clock_timestamp_t get_logical_sample_clock_for_first_sample() const {
    return static_cast<rtp_clock_timestamp_t>(m_logical_clock.count());
  }

  size_t get_size() const {
    return m_size;
  }

  void set_size(size_t size) {
    assert(size < m_data.size());
    m_size = size;
  }

  const AudioFormat &get_format() const {
    return m_format;
  }

  void import(const SampleChunkSingleChannel &c) {
    m_logical_clock = c.m_logical_clock;
    for (size_t i = 0; i < c.m_size; i++) {
      m_data[i] = c.m_data[i];
    }
    m_size = c.m_size;
    m_size = c.m_size;
    m_format = c.m_format;
    m_finished = c.m_finished;
    m_overrun = c.m_overrun;
  }

  static constexpr size_t get_max_size() {
    return MAX_NUMBER_OF_SAMPLES_PER_TIME_PERIOD;
  }

  /** input is a 8/16/24 bit array of samples with len bytes total.
   * We need to copy each sample of the 'intput_channel' into our single
   * channel buffer.
   */
  void import_single_channel(const uint8_t *data, size_t len, channel_index_t input_channel, const AudioFormat &format);

  /** Import samples from interleaved data for this channel.
   * @param src pointer to the start of interleaved data (already offset to this channel's first sample)
   * @param sample_count number of samples to import
   * @param num_channels total number of channels in the interleaved data (used for stride)
   */
  void import_interleaved_8bit(const uint8_t *src, size_t sample_count, channel_index_t num_channels);
  void import_interleaved_16bit(const uint8_t *src, size_t sample_count, channel_index_t num_channels);
  void import_interleaved_24bit(const uint8_t *src, size_t sample_count, channel_index_t num_channels);
  void import_interleaved_32bit(const uint8_t *src, size_t sample_count, channel_index_t num_channels);

private:
  std::chrono::nanoseconds m_logical_clock;

  /** contains the samples for a single audio channel */
  std::array<SampleType, MAX_NUMBER_OF_SAMPLES_PER_TIME_PERIOD> m_data;
  size_t m_size = 0;
  AudioFormat m_format{AudioEncodingID::UNKNOWN, 0};
  bool m_finished = false;
  uint32_t m_overrun = 0;
};

class SampleChunkBlockingLocker {
public:
  SampleChunkBlockingLocker(SampleChunkSingleChannel &c, int owner)
      : m_chunk(c), m_owner(owner) {
    while (!c.try_lock(owner)) {
    }
  }

  SampleChunkBlockingLocker(const SampleChunkBlockingLocker &) = delete;
  SampleChunkBlockingLocker(SampleChunkBlockingLocker &&) = delete;
  SampleChunkBlockingLocker &operator=(const SampleChunkBlockingLocker &) = delete;
  SampleChunkBlockingLocker &&operator=(SampleChunkBlockingLocker &&) = delete;

  ~SampleChunkBlockingLocker() {
    m_chunk.unlock(m_owner);
  }

private:
  SampleChunkSingleChannel &m_chunk;
  int m_owner;
};

class SampleChunkTryLocker {
public:
  SampleChunkTryLocker(SampleChunkSingleChannel &c, int owner)
      : m_chunk(c), m_owner(owner) {
    m_success = c.try_lock(owner);
  }

  SampleChunkTryLocker(const SampleChunkTryLocker &) = delete;
  SampleChunkTryLocker(SampleChunkTryLocker &&) = delete;
  SampleChunkTryLocker &operator=(const SampleChunkTryLocker &) = delete;
  SampleChunkTryLocker &&operator=(SampleChunkTryLocker &&) = delete;

  bool success() const {
    return m_success;
  }

  void unlock() {
    m_success = false;
    m_chunk.unlock(m_owner);
  }

  ~SampleChunkTryLocker() {
    if (m_success) {
      m_chunk.unlock(m_owner);
    }
  }

private:
  SampleChunkSingleChannel &m_chunk;
  bool m_success = false;
  int m_owner;
};

} // namespace audio

#include <cmath>
#include <memory>

#include <audio/DummyAudioService.hpp>
#include <audio/IAudioService.hpp>

namespace audio {

class SineWaveGeneratorAudioService : public DummyAudioService {
public:
  SineWaveGeneratorAudioService(logging::ILogger &logger,
                                AudioProfile &profile,
                                const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                                const std::shared_ptr<model::Device> &device,
                                const std::shared_ptr<ptp::IPtpService> &ptp_service)
      : DummyAudioService(logger, profile, rt_kernel, device, ptp_service) {
  }

  std::string get_name() const override { return "sine"; }
  std::string get_description() const { return "sine-descr"; }

  std::vector<std::string> get_hw_output_channel_names() const { return {"left_out"}; }
  std::vector<std::string> get_hw_input_channel_names() const { return {"left_in"}; }

  error::Error init() override;
  error::Error finish() override;

private:
  double m_seq = 0;
  std::shared_ptr<realtime::PeriodicTask> m_periodic;

  void record(SampleChunkSingleChannel &chunk);
};

} // namespace audio

#include <cstdint>

namespace model {

struct Fraction {
  std::int32_t numerator;
  std::int32_t denominator;
};

} // namespace model

#include <string>

class GUID {
public:
  explicit GUID(const std::string &value)
      : m_value(value) {
  }

  const std::string &to_string() const {
    return m_value;
  }

  bool operator==(const GUID &other) const {
    return m_value == other.m_value;
  }

  static GUID create_new_id();

private:
  std::string m_value;
};

template <>
struct std::formatter<GUID> {
  constexpr auto parse(std::format_parse_context &ctx) {
    return ctx.begin();
  }

  auto format(const GUID &c, std::format_context &ctx) const {
    return std::format_to(ctx.out(), "{}", c.to_string());
  }
};
#pragma once

#include "Fraction.hpp"

namespace model {
using GrainRate = Fraction;
} // namespace model

#include <memory>
#include <string>
#include <vector>

#include <audio/AudioFormat.hpp>

#include "ChannelMapper.hpp"

namespace model {
class Device;

using channel_index_t = audio::channel_index_t;

class DeviceIOChannel {
public:
  explicit DeviceIOChannel(const std::string &id, const std::string &name, const std::string &description, channel_index_t index)
      : m_id(id), m_name(name), m_description(description), m_index(index) {
  }

  std::string get_id() const {
    return m_id;
  }

  std::string get_name() const {
    return m_name;
  }

  std::string get_description() const {
    return m_description;
  }

  channel_index_t get_channel_index() const {
    return m_index;
  }

private:
  std::string m_id;
  std::string m_name;
  std::string m_description;
  channel_index_t m_index;
};

class BaseDeviceInputOutput {
public:
  explicit BaseDeviceInputOutput(const std::string &id, const std::string &name, const std::string &description,
                                 const std::shared_ptr<Device> &device)
      : m_id(id), m_name(name), m_description(description), m_device(device) {
  }

  std::string get_id() const {
    return m_id;
  }

  std::string get_name() const {
    return m_name;
  }

  std::string get_description() const {
    return m_description;
  }

  std::shared_ptr<Device> get_device() const {
    return m_device;
  }

  const std::vector<std::shared_ptr<DeviceIOChannel>> &get_channels() const {
    return m_channels;
  }

  void add_channel(const std::shared_ptr<DeviceIOChannel> &channel) {
    m_channels.push_back(channel);
  }

  std::shared_ptr<DeviceIOChannel> get_channel_by_index(const std::string &channel_id) const {
    if (channel_id.empty()) {
      return nullptr;
    }
    const auto ix_opt = StringUtils::parse_int(channel_id);
    if (!ix_opt.has_value()) {
      return nullptr;
    }
    return get_channel_by_index(ix_opt.value());
  }

  std::shared_ptr<DeviceIOChannel> get_channel_by_index(channel_index_t index) const {
    for (const auto &channel : m_channels) {
      if (channel->get_channel_index() == index) {
        return channel;
      }
    }
    return nullptr;
  }

private:
  std::vector<std::shared_ptr<DeviceIOChannel>> m_channels;
  std::string m_id;
  std::string m_name;
  std::string m_description;
  std::shared_ptr<Device> m_device;
};
} // namespace model

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <audio/AudioFormat.hpp>
#include <audio/IAudioService.hpp>
#include <iuring/IPAddress.hpp>

namespace model {
using channel_index_t = audio::channel_index_t;

class BaseDeviceInputOutput;

struct MappedChannel {
  std::shared_ptr<BaseDeviceInputOutput> audio_service;
  channel_index_t physical_channel;
};

using Mapping = std::pair<MappedChannel, std::optional<MappedChannel>>;

enum class ActivationMode {
  IMMEDIATE,
  SCHEDULED_ABSOLUTE,
  SCHEDULED_RELATIVE
};

struct PendingActivation {
  std::string activation_id;
  ActivationMode mode;

  time_utils::tai::nanoseconds requested_time;  // TAI timestamp
  time_utils::tai::nanoseconds activation_time; // (when it will/did activate)
  time_utils::tai::nanoseconds instantation_time =
      time_utils::tai::get_current_time(); // when it was created

  std::vector<Mapping> action; // The map changes to apply

  time_utils::tai::nanoseconds deadline() const {
    switch (mode) {
    case ActivationMode::SCHEDULED_ABSOLUTE:
      return activation_time;
    case ActivationMode::SCHEDULED_RELATIVE:
      return instantation_time + requested_time;
    default:
      // For IMMEDIATE mode, return the instantation time as the deadline
      return instantation_time;
    }
  }
};

class ChannelMapper {
public:
  ChannelMapper(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel);
  ChannelMapper(const ChannelMapper &) = delete;
  ChannelMapper &operator=(const ChannelMapper &) = delete;
  ChannelMapper(ChannelMapper &&) = delete;
  ChannelMapper &operator=(ChannelMapper &&) = delete;
  ~ChannelMapper() = default;

  /** return the LogicalChannel for a given logical channel index
   */
  std::optional<MappedChannel> get_channel_mapping(const MappedChannel &channel) const {
    for (const auto &existing_mapping : m_map) {
      if (existing_mapping.first.audio_service == channel.audio_service &&
          existing_mapping.first.physical_channel ==
              channel.physical_channel) {
        return existing_mapping.second;
      }
    }
    return std::nullopt;
  }

  /** map logical audio channel to an audio-device
   * for an incoming RTP packet's ip-address
   * and output channel on the output audio device
   */
  void map_channel(
      const MappedChannel &from, const std::optional<MappedChannel> &to) {
    for (auto &existing_mapping : m_map) {
      if (existing_mapping.first.audio_service == from.audio_service &&
          existing_mapping.first.physical_channel ==
              from.physical_channel) {
        existing_mapping.second = to;
        return;
      }
    }
    m_map.emplace_back(from, to);
  }

  /** get all current mappings.
   * The logical channel index is from 0-N where N is the total number of
   * logical channels over all hardware I/O channels.
   */
  const std::vector<Mapping> &get_all_mappings() const {
    return m_map;
  }

  /** Store a pending activation */
  void add_pending_activation(const PendingActivation &activation) {
    m_pending_activations[activation.activation_id] = activation;
  }

  /** Get a pending activation by ID */
  std::optional<PendingActivation> get_pending_activation(
      const std::string &activation_id) const;

  /** Remove a pending activation */
  bool remove_pending_activation(const std::string &activation_id) {
    return m_pending_activations.erase(activation_id) > 0;
  }

  /** Get all pending activations */
  const std::map<std::string, PendingActivation> &
  get_all_pending_activations() const {
    return m_pending_activations;
  }

  /** Set the active activation (the last activation that was applied) */
  void set_active_activation(const PendingActivation &activation) {
    m_active_activation = activation;
  }

  /** Get the active activation (the last activation that was applied) */
  std::optional<PendingActivation> get_active_activation() const {
    return m_active_activation;
  }

  void schedule_activation(const PendingActivation &activation);

private:
  std::shared_ptr<realtime::RealtimeKernel> m_rt_kernel;
  std::vector<Mapping> m_map;

  /** map's activation_ids to PendingActivation objects */
  std::map<std::string, PendingActivation> m_pending_activations;
  std::optional<PendingActivation> m_active_activation;

  std::shared_ptr<realtime::IdleTask> m_idle_task;

  void activate(const PendingActivation &activation);
};
} // namespace model

#include <memory>

#include <nlohmann/json.hpp>

#include <slogger/ILogger.hpp>

#include "DeviceInputs.hpp"
#include "DeviceOutputs.hpp"
#include "GUID.hpp"
#include "Receiver.hpp"
#include "Resource.hpp"
#include "Sender.hpp"
#include "Source.hpp"
#include <Configuration.hpp>

namespace model {
class Node;

class Device : public Resource, public std::enable_shared_from_this<Device> {
public:
  Device(logging::ILogger &logger, const GUID &id,
         const std::string &description, const std::string &label,
         const std::shared_ptr<Node> &node, const std::string &type)
      : Resource(logger, id, description, label), m_type(type), m_node(node) {
  }

  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&) = delete;
  Device &operator=(Device &&) = delete;

  nlohmann::json get_nmos_layout();

  void create_source_senders_and_receivers(const iuring::IPAddress &ip,
                                           http::HttpClientManager &m_http_client_manager,
                                           const std::shared_ptr<audio::IRTP_Service> &rtp_service);

  void set_audio_device(
      const std::shared_ptr<audio::IAudioService> &audio_device) {
    m_audio_device = audio_device;
  }

  const std::string &get_type() const {
    return m_type;
  }

  void add_input(const std::shared_ptr<DeviceInput> &input) {
    m_inputs.push_back(input);
  }

  void add_output(const std::shared_ptr<DeviceOutput> &output) {
    m_outputs.push_back(output);
  }

  const std::vector<std::shared_ptr<DeviceOutput>> &
  get_channel_outputs() const {
    return m_outputs;
  }

  const std::vector<std::shared_ptr<DeviceInput>> &get_channel_inputs() const {
    return m_inputs;
  }

  /** returns nullptr on error */
  std::shared_ptr<Sender> get_sender(const GUID &id) const;

  /** returns nullptr on error */
  std::shared_ptr<Receiver> get_receiver(const GUID &id) const;

  const std::vector<std::shared_ptr<Sender>> &get_senders() const {
    return m_senders;
  }

  const std::vector<std::shared_ptr<Receiver>> &get_receivers() const {
    return m_receivers;
  }

  size_t num_senders() const {
    return m_senders.size();
  }

  size_t num_receivers() const {
    return m_receivers.size();
  }

  std::vector<std::string> get_sender_ids() const;
  std::vector<std::string> get_receiver_ids() const;

  const std::vector<std::shared_ptr<Source>> get_sources() const {
    return m_sources;
  }

  void add_source(const std::shared_ptr<Source> &source) {
    m_sources.push_back(source);
  }

  void add_sender(const std::shared_ptr<Sender> &s) {
    m_senders.push_back(s);
  }

  void add_receiver(const std::shared_ptr<Receiver> &s) {
    m_receivers.push_back(s);
  }

  const std::shared_ptr<Node> &get_node() const {
    return m_node;
  }

  const std::shared_ptr<audio::IAudioService> &get_audio_device() const {
    return m_audio_device;
  }

  void for_each_source(
      const std::function<void(const std::shared_ptr<Source> &)> &func) const {
    for (const auto &ds : m_sources) {
      func(ds);
    }
  }

  void add_flow(const std::shared_ptr<model::Flow> &flow) {
    assert(m_sources.size() > 0);

    m_sources[0]->add_flow(flow);
  }

private:
  std::shared_ptr<audio::IAudioService> m_audio_device;

  std::string m_type;
  std::shared_ptr<Node> m_node;

  std::vector<std::shared_ptr<Source>> m_sources;

  // current sender:
  std::vector<std::shared_ptr<Sender>> m_senders;

  // current receiver:
  std::vector<std::shared_ptr<Receiver>> m_receivers;

  // hardware supported:
  std::vector<std::shared_ptr<DeviceInput>> m_inputs;

  // hardware supported:
  std::vector<std::shared_ptr<DeviceOutput>> m_outputs;

  uint32_t m_unique_input_id = 0;
  uint32_t m_unique_output_id = 0;
};

} // namespace model

#include <string>

#include <nlohmann/json.hpp>

#include "BaseDeviceInputOutput.hpp"

namespace model {
class DeviceInput : public BaseDeviceInputOutput {
public:
  DeviceInput(const std::string &id, const std::string &name, const std::string &description, const std::shared_ptr<Device> &device)
      : BaseDeviceInputOutput(id, name, description, device) {
  }

  uint8_t get_block_size() const { return 8; }
  bool is_reordering_supported() const { return true; }

  nlohmann::json get_nmos_layout();
};
} // namespace model

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "BaseDeviceInputOutput.hpp"

namespace model {
class Device;

class DeviceOutput : public BaseDeviceInputOutput {
public:
  DeviceOutput(const std::string &id, const std::string &name, const std::string &description,
               const std::shared_ptr<Device> &device)
      : BaseDeviceInputOutput(id, name, description, device) {
  }

  nlohmann::json get_nmos_layout();
};

} // namespace model

#include "GUID.hpp"
#include "GrainRate.hpp"
#include "Resource.hpp"

#include <iuring/IPAddress.hpp>

#include <audio/AudioFormat.hpp>

#include "StreamInfo.hpp"

namespace model {
class Source;

class Flow : public Resource {
public:
  Flow(logging::ILogger &logger, const GUID id,
       const std::string &description, const std::string &label,
       const std::shared_ptr<Flow> &parent_flow,
       const std::shared_ptr<Source> &parent_source)
      : Resource(logger, id, description, label), m_parent_flow(parent_flow), m_parent_source(parent_source) {
  }

  GrainRate nmos_grain() const;

  std::shared_ptr<Flow> get_parent_flow() const {
    return m_parent_flow;
  }
  std::shared_ptr<Source> get_parent_source() const {
    return m_parent_source;
  }

  void operator=(const Flow &other) {
    m_owner = other.m_owner;
    m_owner_timestamp = other.m_owner_timestamp;
    m_owner_hash = other.m_owner_hash;
    m_owner_tech = other.m_owner_tech;
    m_owner_protocol = other.m_owner_protocol;
    m_owner_ip_address = other.m_owner_ip_address;
    m_session_info = other.m_session_info;
    m_channel_names = other.m_channel_names;
    m_stream_tech = other.m_stream_tech;
    m_stream_protocol = other.m_stream_protocol;
    m_stream_address = other.m_stream_address;
    m_streams = other.m_streams;
  }

  void set_owner(const std::string &owner, const std::string &timestamp,
                 const std::string &hash, const std::string &tech,
                 const std::string &protocol, const iuring::IPAddress &ip_address) {
    m_owner = owner;
    m_owner_timestamp = timestamp;
    m_owner_hash = hash;
    m_owner_tech = tech;
    m_owner_protocol = protocol;
    m_owner_ip_address = ip_address;
  }

  void set_session_info(const std::string &si) {
    m_session_info = si;
  }

  void add_channel(const std::string &ch_name) {
    m_channel_names.push_back(ch_name);
  }

  const std::vector<std::string> &get_channel_names() const {
    return m_channel_names;
  }

  iuring::IPAddress get_owner_ip_address() const {
    return m_owner_ip_address;
  }

  const iuring::IPAddress &get_stream_ip_address() const {
    return m_stream_address;
  }

  std::vector<int> get_stream_ports() const {
    std::vector<int> ret;
    for (const auto &it : m_streams) {
      const auto port =
          std::underlying_type_t<iuring::SocketPortID>(it.get_port());
      ret.push_back(port);
    }
    return ret;
  }

  void set_address_info(const std::string &stream_tech,
                        const std::string &stream_protocol,
                        const iuring::IPAddress &stream_address) {
    m_stream_tech = stream_tech;
    m_stream_protocol = stream_protocol;
    m_stream_address = stream_address;
  }

  const std::vector<StreamInfo> &get_streams() const {
    return m_streams;
  }

  void add_stream(const std::string &stream_type,
                  const iuring::SocketPortID port, const std::string &protocol,
                  const int idx) {
    m_streams.push_back(
        StreamInfo{stream_type, port, protocol, idx, get_logger()});
  }

  bool set_attribute_for_last_stream(
      const std::string &key, const std::string &value) {
    if (m_streams.empty()) {
      LOG_ERROR(
          get_logger(), "no stream to associate attribute with: {}", key);
      return false;
    }
    m_streams.back().add_map(key, value);
    return true;
  }

  bool is_valid() const {
    if (m_streams.empty())
      return false;
    if (m_channel_names.empty())
      return false;
    return true;
  }

  bool operator==(const Flow &f) const {
    return (m_owner == f.m_owner) &&
           (m_stream_address == f.m_stream_address);
  }

private:
  std::shared_ptr<Flow> m_parent_flow;
  std::shared_ptr<Source> m_parent_source;

  std::string m_owner;
  std::string m_owner_timestamp;
  std::string m_owner_hash;
  std::string m_owner_tech;
  std::string m_owner_protocol;
  iuring::IPAddress m_owner_ip_address;
  std::string m_session_info;

  // i=2 channels: Left, Right
  std::vector<std::string> m_channel_names;

  // IN IP4 239.69.199.207/32
  std::string m_stream_tech;
  std::string m_stream_protocol;
  iuring::IPAddress m_stream_address;

  /*
      m=audio 5004 RTP/AVP 97
      a=rtpmap:97 L24/48000/2
      a=ptime:1
      a=ts-refclk:ptp=IEEE1588-2008:00-0C-29-FF-FE-E3-07-9F:0
      a=mediaclk:direct=0
  */
  std::vector<StreamInfo> m_streams;
};
} // namespace model
#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <slogger/Error.hpp>
#include <slogger/ILogger.hpp>
#include <slogger/StringUtils.hpp>
#include <slogger/TimeUtils.hpp>

#include <iuring/IOUringInterface.hpp>
#include <iuring/NetworkAdapter.hpp>

#include <urtsched/IService.hpp>

#include <audio/AudioProfile.hpp>

#include "domainmodel/GUID.hpp"
#include "nmos/nmos_codegen_types.hpp"

#include <aoip/RavennaService.hpp>
#include <audio/RTPAudioDevice.hpp>

namespace audio {
class IAlsaManager;
}

namespace network {
class INetwork;
}

namespace ptp {
class IPtpService;
}

namespace mdns {
class MDNS_Service;
}

namespace service {
class ServiceBus;
}

namespace realtime {
class RealtimeKernel;
}
namespace NMOS {
class NMOS_Service;
}

#include <http/HttpClientManager.hpp>

#include "Resource.hpp"

#include "Device.hpp"
#include "Receiver.hpp"
#include "Sender.hpp"
#include "aoip/IRTP_Service.hpp"
#include <domainmodel/ChannelMapper.hpp>

namespace model {
class Node : public Resource, public std::enable_shared_from_this<Node> {
public:
  Node(logging::ILogger &logger, const GUID &id,
       const std::string &description, const std::string &label,
       const std::shared_ptr<iuring::IOUringInterface> &network,
       iuring::NetworkAdapter &adapter,
       http::HttpClientManager &http_client_manager,
       settings::Configuration &config,
       const std::shared_ptr<ptp::IPtpService> &ptp_service,
       model::ChannelMapper &mapper)
      : Resource(logger, id, description, label), m_io(network), m_adapter(adapter), m_http_client_manager(http_client_manager), m_config(config), m_ptp_service(ptp_service), m_mapper(mapper) {
  }

  nlohmann::json get_nmos_layout();

  bool has_flow(const std::shared_ptr<model::Flow> &flow) const {
    for (const auto &f : get_flows()) {
      if (*f == *flow) {
        return true;
      }
    }
    return false;
  }

  void remove_flow(const std::shared_ptr<model::Flow> &flow);
  void add_flow(const std::shared_ptr<model::Flow> &flow);

  model::ChannelMapper &get_mapper() {
    return m_mapper;
  }

  std::string get_nmos_gmid();
  std::string get_nmos_service_url();

  std::shared_ptr<Source> get_local_source() const {
    return m_devices.empty() ? nullptr : m_devices[0]->get_sources().empty() ? nullptr
                                                                             : m_devices[0]->get_sources()[0];
  }

  std::shared_ptr<audio::RTPReceiverAudioService>
  get_local_rtp_playback_device() const {
    return m_local_rtp_playback_device;
  }

  void set_local_rtp_playback_device(
      const std::shared_ptr<audio::RTPReceiverAudioService> &dev) {
    m_local_rtp_playback_device = dev;
  }

  iuring::SocketPortID get_nmos_port() const {
    return m_config.http.web_port;
  }

  /** create the model by seeing which audiio devices exist (instead of
   * read()) */
  void create(const std::shared_ptr<audio::IRTP_Service> &rtp_service,
              const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
              audio::IAlsaManager &alsa_manager);

  /** read the model from a file instead of create() */
  void read(const std::string &filename,
            const std::shared_ptr<audio::IRTP_Service> &rtp_service,
            const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel);

  const std::vector<std::shared_ptr<Device>> &get_devices() const {
    return m_devices;
  }

  std::vector<std::string> get_api_versions() const {
    return {"v1.3"};
  }

  std::vector<std::shared_ptr<Sender>> get_senders() const;
  std::vector<std::shared_ptr<Receiver>> get_receivers() const;
  std::vector<std::shared_ptr<Source>> get_sources() const;
  std::vector<std::shared_ptr<Flow>> get_flows() const;

  /** returns nullptr on error */
  std::shared_ptr<Sender> get_sender(const GUID &id) const;
  /** returns nullptr on error */
  std::shared_ptr<Receiver> get_receiver(const GUID &id) const;

  size_t num_senders() const;
  size_t num_receivers() const;

  std::shared_ptr<Flow> find_flow(const GUID &id) const;
  std::shared_ptr<Source> find_source(const GUID &id) const;
  std::shared_ptr<Device> find_device(const GUID &id) const;
  std::shared_ptr<Sender> find_sender(const GUID &id) const;
  std::shared_ptr<Receiver> find_receiver(const GUID &id) const;

  http::hostname_t get_hostname() const {
    return {m_adapter.get_hostname()};
  }

  [[nodiscard]] error::Error init(logging::ILogger &logger,
                                  const std::shared_ptr<audio::IRTP_Service> &m_rtp_service,
                                  service::ServiceBus &bus,
                                  const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                                  const std::shared_ptr<iuring::IOUringInterface> &network,
                                  const std::shared_ptr<mdns::MDNS_Service> &mdns_server,
                                  iuring::NetworkAdapter &adapter,
                                  iuring::ISocketFactory &socket_factory);

  std::shared_ptr<iuring::IOUringInterface> get_io() {
    return m_io;
  }

  iuring::NetworkAdapter &get_adapter() {
    return m_adapter;
  }

  settings::Configuration &get_config() {
    return m_config;
  }

  const std::shared_ptr<ptp::IPtpService> &get_ptp_service() const {
    return m_ptp_service;
  }

  std::optional<std::shared_ptr<model::DeviceInput>> find_input_by_id(
      const std::string &input_id);
  std::optional<std::shared_ptr<model::DeviceOutput>> find_output_by_id(
      const std::string &output_id);

  const std::shared_ptr<audio::RavennaService> &get_ravenna_service() const {
    return m_ravenna;
  }

private:
  std::vector<std::shared_ptr<Device>> m_devices;
  std::shared_ptr<iuring::IOUringInterface> m_io;
  iuring::NetworkAdapter &m_adapter;
  http::HttpClientManager &m_http_client_manager;
  settings::Configuration &m_config;
  std::shared_ptr<ptp::IPtpService> m_ptp_service;
  model::ChannelMapper &m_mapper;
  std::shared_ptr<audio::RTPReceiverAudioService> m_local_rtp_playback_device;
  std::shared_ptr<audio::RavennaService> m_ravenna;

  void create_RTP_device_model(const iuring::IPAddress &ip,
                               const std::shared_ptr<audio::IRTP_Service> &rtp_service,
                               const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel);

  void create_sinewave_device_model(const iuring::IPAddress &ip,
                                    const std::shared_ptr<audio::IRTP_Service> &rtp_service,
                                    const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel);

  void create_ALSA_device_model(const iuring::IPAddress &ip,
                                const std::shared_ptr<audio::IRTP_Service> &rtp_service,
                                const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                                audio::IAlsaManager &alsa_manager);

  void read_device(const nlohmann::json &device_json,
                   const std::shared_ptr<audio::IRTP_Service> &rtp_service,
                   const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel);

  void read_device_audio_inputs(const nlohmann::json &device_json,
                                const std::shared_ptr<Device> &device);
  void read_device_audio_outputs(const nlohmann::json &device_json,
                                 const std::shared_ptr<Device> &device);
  std::shared_ptr<Flow> read_device_sources(const nlohmann::json &device_json,
                                            const std::shared_ptr<Device> &device);
  void read_device_senders(const nlohmann::json &device_json,
                           const std::shared_ptr<Device> &device);
  void read_device_receivers(const nlohmann::json &device_json,
                             const std::shared_ptr<Device> &device,
                             const std::shared_ptr<Flow> &flow,
                             const std::shared_ptr<audio::IRTP_Service> &rtp_service);
};

} // namespace model

#include <functional>
#include <memory>
#include <string>

#include <http/HttpClientManager.hpp>
#include <http/URL.hpp>

#include "GUID.hpp"
#include "URN.hpp"

#include "Resource.hpp"

#include <aoip/IRTP_Service.hpp>
#include <domainmodel/Flow.hpp>

namespace model {
class Device;
class Flow;

using subscribe_handler_t = std::function<void(const error::Error &)>;

class Receiver : public Resource {
public:
  Receiver(logging::ILogger &logger, const GUID &id,
           const std::string &description, const std::string &label,
           const std::shared_ptr<Device> &device,
           const std::shared_ptr<Flow> &flow,
           http::HttpClientManager &http_client_manager,
           const std::shared_ptr<audio::IRTP_Service> &rtp_service)
      : Resource(logger, id, description, label), m_device(device), m_flow(flow), m_http_client_manager(http_client_manager), m_rtp_service(rtp_service) {
  }

  Receiver(const Receiver &) = delete;
  Receiver &operator=(const Receiver &) = delete;
  Receiver(Receiver &&) = delete;
  Receiver &operator=(Receiver &&) = delete;

  nlohmann::json get_nmos_layout();

  bool get_master_enable() const { return m_master_enable; }

  std::optional<URN> get_transport_type_urn_proto() const { return m_transport_type_urn_proto; }
  std::optional<URN> get_transport_type_urn_mcast() const { return m_transport_type_urn_mcast; }
  std::optional<http::URL> get_manifest_href() const { return m_manifest_href; }

  /** calls handler when subscription completes */
  void subscribe(const GUID &sender_id, const http::URL &href_manifest, subscribe_handler_t handler);
  error::Error unsubscribe();

  std::shared_ptr<Device> get_device() const {
    return m_device;
  }

  std::shared_ptr<Flow> get_flow() const {
    return m_flow;
  }

  void set_flow(const std::shared_ptr<Flow> &flow) {
    m_flow = flow;
  }

  bool is_active() const {
    return m_active;
  }

  bool has_constraints() {
    return false;
  }

  bool is_staged() const {
    return m_staged;
  }

  bool has_transport_file() const {
    return m_manifest_href.has_value();
  }

  bool has_transport_type() const {
    return m_transport_type_urn_proto.has_value();
  }

  const std::string &get_sdp_file() const { return m_sdp_file; }

  void set_sender_id(const GUID &sender_id) { m_sender_id = sender_id; }
  const std::optional<GUID> &get_sender_id() const { return m_sender_id; }

private:
  bool m_master_enable = true;
  bool m_active = true;
  bool m_staged = false;

  std::optional<GUID> m_sender_id;

  std::optional<URN> m_transport_type_urn_mcast = URN("urn:x-nmos:transport:rtp.mcast");
  std::optional<URN> m_transport_type_urn_proto = URN("urn:x-nmos:transport:rtp");
  std::optional<http::URL> m_manifest_href = std::nullopt;

  std::string m_sdp_file;

  std::shared_ptr<Device> m_device;
  std::shared_ptr<Flow> m_flow;

  http::HttpClientManager &m_http_client_manager;

  std::shared_ptr<audio::IRTP_Service> m_rtp_service;
};

} // namespace model

#include <string>

#include <slogger/ILogger.hpp>
#include <slogger/TimeUtils.hpp>

#include <nmos/nmos_codegen_types.hpp>

#include "GUID.hpp"

namespace model {
class Resource {
public:
  Resource(logging::ILogger &logger, const GUID &id, const std::string &description, const std::string &label)
      : m_logger(logger), m_id(id), m_description(description), m_label(label) {}

  virtual ~Resource() = default;

  void set_id(const GUID &id) {
    m_id = id;
  }

  http::KeyValueSet get_tags() const {
    return {};
  }

  logging::ILogger &get_logger() const {
    return m_logger;
  }

  const GUID &get_id() const {
    return m_id;
  }

  const std::string &get_description() const {
    return m_description;
  }

  const std::string &get_label() const {
    return m_label;
  }

  std::string get_TAI_timestamp_of_last_change() {
    // seconds:nanoseconds
    return m_last_change_timepoint.to_TAI_timestamp();
  }

  void notify_change() {
    m_last_change_timepoint = time_utils::tai::get_current_time();
  }

private:
  logging::ILogger &m_logger;
  GUID m_id;
  std::string m_description;
  std::string m_label;
  time_utils::tai::nanoseconds m_last_change_timepoint = time_utils::tai::get_current_time();
};

} // namespace model

#include "GUID.hpp"
#include "Resource.hpp"
#include "URN.hpp"

namespace model {
class Device; // forward decl

class Sender : public Resource {
public:
  Sender(logging::ILogger &logger, const GUID &id,
         const std::string &description, const std::string &label,
         const std::shared_ptr<Device> &device)
      : Resource(logger, id, description, label), m_device(device) {
  }

  Sender(const Sender &) = delete;
  Sender &operator=(const Sender &) = delete;
  Sender(Sender &&) = delete;
  Sender &operator=(Sender &&) = delete;

  nlohmann::json get_nmos_layout();

  std::shared_ptr<Device> get_device() const {
    return m_device;
  }

  std::optional<URN> get_transport_type_urn_proto() const {
    return m_transport_type_urn_proto;
  }
  std::optional<URN> get_transport_type_urn_mcast() const {
    return m_transport_type_urn_mcast;
  }

  bool is_active() const {
    return m_active;
  }

  bool has_constraints() {
    return false;
  }

  bool is_staged() const {
    return m_staged;
  }

  bool has_transport_file() const {
    return true;
  }

  bool has_transport_type() const {
    return true;
  }

  bool master_enable() const {
    return m_master_enable;
  }

  std::string get_sdp_data() const;

  void set_receiver_id(const GUID &receiver_id) {
    m_receiver_id = receiver_id;
  }
  const std::optional<GUID> &get_receiver_id() const {
    return m_receiver_id;
  }

private:
  std::optional<GUID> m_receiver_id;
  bool m_active = true;
  bool m_staged = false;
  bool m_master_enable = true;
  std::shared_ptr<Device> m_device;
  std::optional<URN> m_transport_type_urn_mcast =
      URN("urn:x-nmos:transport:rtp.mcast");
  std::optional<URN> m_transport_type_urn_proto =
      URN("urn:x-nmos:transport:rtp");
};
} // namespace model
#pragma once

#include <memory>
#include <string>

#include "Resource.hpp"

#include "Flow.hpp"
#include "GUID.hpp"
#include "GrainRate.hpp"

#include "DeviceInputs.hpp"
#include "DeviceOutputs.hpp"

namespace model {
class Device;

class Source : public Resource, public std::enable_shared_from_this<Source> {
public:
  Source(logging::ILogger &logger, const GUID &id,
         const std::string &description, const std::string &label,
         const std::shared_ptr<Device> &parent_device)
      : Resource(logger, id, label, description), m_parent_device(parent_device) {
  }

  nlohmann::json get_nmos_layout();
  GrainRate nmos_grain() const;

  const std::vector<std::shared_ptr<Flow>> &get_flows() const {
    return m_flows;
  }

  std::shared_ptr<Device> get_parent_device() const {
    return m_parent_device;
  }

  void remove_flow(const std::shared_ptr<model::Flow> &flow) {
    m_flows.erase(std::remove_if(m_flows.begin(), m_flows.end(),
                                 [&](const std::shared_ptr<model::Flow> &f) {
                                   return *f == *flow;
                                 }),
                  m_flows.end());
  }

  void add_flow(const std::shared_ptr<model::Flow> &flow) {
    m_flows.push_back(flow);
  }

private:
  std::vector<std::shared_ptr<Flow>> m_flows;
  std::shared_ptr<Device> m_parent_device;
};
} // namespace model
#pragma once

#include <map>
#include <string>

#include <iuring/SocketPortID.hpp>
#include <optional>
#include <slogger/ILogger.hpp>

#include <audio/AudioFormat.hpp>

namespace model {
class StreamInfo {
private:
  std::string m_stream_type; // audio / video
  iuring::SocketPortID m_port;
  std::string m_protocol; // 'IP4' / 'IP6'
  int m_idx;              // audio attributes
  std::map<std::string, std::string> m_map;
  logging::ILogger &m_logger;

public:
  StreamInfo(const std::string &stream_type, iuring::SocketPortID port,
             std::string protocol, int idx, logging::ILogger &logger)
      : m_stream_type(stream_type), m_port(port), m_protocol(protocol), m_idx(idx), m_logger(logger) {
  }

  void operator=(const StreamInfo &other) {
    m_stream_type = other.m_stream_type;
    m_port = other.m_port;
    m_protocol = other.m_protocol;
    m_idx = other.m_idx;
    m_map = other.m_map;
  }

  logging::ILogger &get_logger() const {
    return m_logger;
  }

  void add_map(const std::string &key, const std::string &value) {
    m_map[key] = value;
  }

  iuring::SocketPortID get_port() const {
    return m_port;
  }
  std::string get_stream_type() const {
    return m_stream_type;
  }
  std::string get_protocol() const {
    return m_protocol;
  }

  bool is_ipv4() const {
    return m_protocol == "IP4";
  }

  bool is_ipv6() const {
    return m_protocol == "IP6";
  }

  std::optional<audio::AudioFormat> get_audio_format() const;
};

} // namespace model
#pragma once

class URN {
public:
  explicit URN(const std::string &value)
      : m_value(value) {
  }

  const std::string &to_string() const {
    return m_value;
  }

  bool operator==(const URN &other) const {
    return m_value == other.m_value;
  }

private:
  std::string m_value;
};
#pragma once

#include <queue>
#include <string>

#include <slogger/ILogger.hpp>
#include <slogger/ITimer.hpp>

#include <iuring/IOUringInterface.hpp>
#include <iuring/IPAddress.hpp>

#include <urtsched/RealtimeKernel.hpp>
#include <urtsched/Service.hpp>

#include <mdns/MDNS_NMOS_HTTP_Handler.hpp>
#include <mdns/MDNS_Service.hpp>

#include <http/HttpClient.hpp>
#include <http/HttpClientManager.hpp>
#include <http/HttpServer.hpp>

#include "gen/all_endpoints-ChannelMappingAPI.hpp"
#include "gen/all_endpoints-ConnectionAPI.hpp"
#include "gen/all_endpoints-EventsAPI.hpp"
#include "gen/all_endpoints-NodeAPI.hpp"
#include "gen/all_endpoints-RegistrationAPI.hpp"

#include <domainmodel/Node.hpp>

#include <Configuration.hpp>

namespace NMOS {
class NMOS_Service : public service::Service, public mdns::INMOS_Service {
public:
  NMOS_Service(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
               const std::shared_ptr<iuring::IOUringInterface> &network,
               logging::ILogger &logger, http::HttpServer &http_server,
               const std::shared_ptr<model::Node> &model,
               const std::shared_ptr<mdns::MDNS_Service> &mdns_server,
               iuring::NetworkAdapter &adapter, const settings::Configuration &config,
               http::HttpClientManager &http_client_manager);

  [[nodiscard]] error::Error init();
  [[nodiscard]] error::Error finish() override {
    return error::Error::OK;
  }

  void start_registration(
      const iuring::IPAddress &ip, std::optional<uint16_t> port);

  size_t num_self() const {
    return 1;
  }
  size_t num_source() const {
    return 1;
  }
  size_t num_flows() const {
    return 1;
  }
  size_t num_devices() const {
    return 1;
  }
  size_t num_senders() const {
    return m_model->num_senders();
  }
  size_t num_receivers() const {
    return m_model->num_receivers();
  }

private:
  std::shared_ptr<realtime::RealtimeKernel> m_rt_kernel;
  http::HttpServer &m_http_server;

  // generated code:
  NodeAPI::AllEndpoints m_node_api_endpoints;
  ConnectionAPI::AllEndpoints m_connection_api_endpoints;
  ChannelMappingAPI::AllEndpoints m_channel_mapping_api_endpoints;
  EventsAPI::AllEndpoints m_events_api_endpoints;

  std::shared_ptr<mdns::MDNS_Service> m_mdns_server;

  std::shared_ptr<model::Node> m_model;
  std::shared_ptr<iuring::IOUringInterface> m_io;
  iuring::NetworkAdapter &m_adapter;
  const settings::Configuration &m_config;
  http::HttpClientManager &m_http_client_manager;

  class Registration {
  public:
    enum class Status {
      IN_PROGRESS,
      ACTIVE
    };

    Registration(const iuring::IPAddress &addr, Status status)
        : m_registration_service_addr(addr), m_status(status) {
    }

    void set_periodic(const std::shared_ptr<realtime::PeriodicTask> &periodic) {
      m_periodic = periodic;
      m_status = Status::ACTIVE;
    }

    void reset_timestamp() {
      m_last_failure_timestamp = std::nullopt;
    }

    bool timer_is_running() const { return m_last_failure_timestamp.has_value(); }

    void start_timer(time_utils::ITimer &timer, const std::chrono::microseconds &timeout) {
      m_last_failure_timestamp.emplace(timer, timeout);
    }

    bool elapsed() const {
      if (m_last_failure_timestamp.has_value()) {
        return m_last_failure_timestamp.value().elapsed();
      }
      return false;
    }

    Status get_status() const { return m_status; }
    bool is_active() const { return m_status == Status::ACTIVE; }
    bool is_in_progress() const { return m_status == Status::IN_PROGRESS; }

    std::shared_ptr<realtime::PeriodicTask> get_periodic() { return m_periodic; }

  private:
    std::optional<time_utils::Timeout> m_last_failure_timestamp;
    iuring::IPAddress m_registration_service_addr;
    std::shared_ptr<realtime::PeriodicTask> m_periodic;
    Status m_status;
  };

  std::map<iuring::IPAddress, Registration> m_registrations;

  iuring::NetworkAdapter &get_adapter() {
    return m_adapter;
  }

  std::shared_ptr<iuring::IOUringInterface> get_io() {
    return m_io;
  }

  void handle_registration(const iuring::IPAddress &dest);
  void register_node(const iuring::IPAddress &dest);
  void register_devices(const iuring::IPAddress &dest);
  void register_senders(const iuring::IPAddress &dest);
  void register_sources(const iuring::IPAddress &dest);
  void register_flows(const iuring::IPAddress &dest);
  void register_receivers(const iuring::IPAddress &dest);

  void handle_get_file(const std::string &endpoint,
                       const std::string &payload, const http::URLParameters &params,
                       http::reply_handler_t reply_handler);

  void start_keep_alive(const iuring::IPAddress &ip);
  void send_keep_alive_msg(const iuring::IPAddress &ip);
  void check_needs_keep_alive_removal(
      const iuring::IPAddress &ip, bool saw_error);

  bool have_already_registered(const iuring::IPAddress &ip);

  /** register the Nth device and only then start registering the sources */
  void do_register_device(const iuring::IPAddress &dest, int ix);
  void do_register_source(const iuring::IPAddress &dest, int ix);
  void do_register_flow(const iuring::IPAddress &dest, int ix);
  void do_register_sender(const iuring::IPAddress &dest, int ix);
  void do_register_receiver(const iuring::IPAddress &dest, int ix);
};

} // namespace NMOS
#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace http {

class EmptyObject {
};

#define EMPTY_DECL(CODE)     \
  class EmptyObject_##CODE { \
  };

EMPTY_DECL(200)
EMPTY_DECL(202)
EMPTY_DECL(204)

EMPTY_DECL(307)

EMPTY_DECL(403)
EMPTY_DECL(404)
EMPTY_DECL(405)
EMPTY_DECL(409)
EMPTY_DECL(423)

EMPTY_DECL(500)

class hostname_t {
public:
  hostname_t() {}
  hostname_t(const std::string &value) : value(value) {}

  std::string value;
};

class ipv4_t {
public:
  ipv4_t() {}
  ipv4_t(const std::string v) : value(v) {}
  std::string value;
};

class ipv6_t {
public:
  ipv6_t() {}
  ipv6_t(const std::string v) : value(v) {}
  std::string value;
};

class null_t {
public:
  null_t() {}
  void *dummy = nullptr;
};

class RegEx {
public:
  RegEx() {}

  std::string pattern;
  std::string value;
};

using JsonObject = EmptyObject;
using EmptyType = EmptyObject;

using RegExString = std::string;

using KeyValueSet =
    std::map<std::string,
             std::vector<std::variant<
                 std::string,
                 int64_t,
                 null_t,
                 std::map<std::string,
                          std::variant<std::string,
                                       null_t,
                                       int64_t>>>>>;

} // namespace http

#include <chrono>
#include <cstring>
#include <ctype.h>
#include <string>

#include <netinet/in.h>
#include <sys/types.h>

#include <audio/NetworkProfiles.hpp>

#include <iuring/IOUringInterface.hpp>

namespace ptp::v1 {
using sequence_id_t = uint16_t;

enum class PtpTech {
  ETHERNET,
  UNKNOWN
};

static constexpr uint8_t GRANDMASTER_ETHERNET_TECH = 1;

struct __attribute__((packed)) uuid_type_t {
  uint8_t data[6]{};

  std::string to_string() const {
    std::string ret;
    for (int i = 0; i < 6; i++) {
      int v = data[i];
      char buf[32];
      sprintf(buf, "%.2x", v);
      if (i > 0)
        ret += ':';
      ret += buf;
    }
    return ret;
  }
};

struct __attribute__((packed)) id_type_t {
  uint8_t data[4]{};

  void set(const char *s) {
    memcpy(data, s, sizeof(data));
  }
};

static inline std::string uuid_to_string(const uuid_type_t &uuid) {
  std::string ret;
  const char *sep = "";
  for (std::size_t i = 0; i < sizeof(uuid.data); i++) {
    const auto val = uuid.data[i];
    char buf[8];
    sprintf(buf, "%.2x", val);
    ret += sep;
    ret += buf;
    sep = ":";
  }
  return ret;
}

struct __attribute__((packed)) TimestampV1 {
private:
  uint32_t m_seconds;
  uint32_t m_nanos;

public:
  TimestampV1(const std::chrono::nanoseconds &nanos) {
    const uint64_t mm = (nanos.count());
    const uint64_t mod_nanos = mm % 1000000000ull;
    const uint64_t mod_secs = mm / 1000000000ull;

    m_seconds = ntohl(mod_secs);
    m_nanos = ntohl(mod_nanos);
  }

  std::chrono::nanoseconds get_micros() const {
    const auto secs_as_micros = std::chrono::seconds(get_secs());
    const auto nanos_as_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::nanoseconds(get_nanos()));
    return secs_as_micros + nanos_as_micros;
  }

  uint32_t get_secs() const {
    return ntohl(m_seconds);
  }

  uint32_t get_nanos() const {
    return ntohl(m_nanos);
  }
};

struct __attribute__((packed)) DelayResponseV1 {
private:
  TimestampV1 m_delay_receipt_timestamp;
  uint8_t dummy = 0;
  uint8_t m_requestingSourceCommunicationTechnology =
      GRANDMASTER_ETHERNET_TECH;
  uuid_type_t m_requestingSourceUuid{};
  uint16_t m_requestingSourcePortId = 0;
  sequence_id_t m_requestingSourceSequenceId = 0;

public:
  DelayResponseV1(const TimestampV1 &ts, const uuid_type_t &id, uint16_t port,
                  sequence_id_t seq)
      : m_delay_receipt_timestamp(ts), m_requestingSourceUuid(id), m_requestingSourcePortId(htons(port)), m_requestingSourceSequenceId(htons(seq)) {
  }

  uint16_t get_requestingSourcePortId() const {
    return htons(m_requestingSourcePortId);
  }

  sequence_id_t get_requestingSourceSequenceId() const {
    return htons(m_requestingSourceSequenceId);
  }

  uint8_t get_requestingSourceCommunicationTechnology() const {
    return m_requestingSourceCommunicationTechnology;
  }
};

struct __attribute__((packed)) FollowMessageV1 {
private:
  uint16_t reserved = 0;
  sequence_id_t assoc_sequence_id; // 2 33
  TimestampV1 precisionOriginTimestamp;

public:
  FollowMessageV1(const TimestampV1 &ts, sequence_id_t seq_id)
      : precisionOriginTimestamp(ts) {
    assoc_sequence_id = ntohs(seq_id);
  }

  sequence_id_t get_assoc_sequence_id() const {
    return ntohs(assoc_sequence_id);
  }

  const TimestampV1 &get_precisionOriginTimestamp() const {
    return precisionOriginTimestamp;
  }
};

struct __attribute__((packed)) GrandMasterInfo {
  uint8_t grandmaster_tech = GRANDMASTER_ETHERNET_TECH;
  uuid_type_t grandmaster_clock_uuid{};
  uint16_t grandmaster_port_id = 0;
  sequence_id_t grandmaster_seq_id = 0;
  uint8_t dummy2[3]{};
  uint8_t grandmaster_clock_stratum = 160;
  id_type_t grandmaster_clock_id{};
  uint8_t dummy3[2]{};
  uint16_t grandmaster_clock_variance = 30000;
  uint8_t dummy4 = 0;
  uint8_t grandmaster_preferred = 0;
  uint8_t dummy5 = 0;
  uint8_t grandmaster_is_boundary_clock = 1;

  std::string get_uuid() const {
    return uuid_to_string(grandmaster_clock_uuid);
  }

  /** @return true if 'this' better than 'info
   */
  bool is_better_than(const GrandMasterInfo &info) const {
    if (grandmaster_preferred > info.grandmaster_preferred) {
      return true;
    }
    if (grandmaster_is_boundary_clock > info.grandmaster_is_boundary_clock) {
      return true;
    }
    if (grandmaster_clock_stratum > info.grandmaster_clock_stratum) {
      return true;
    }
    if (grandmaster_clock_variance > info.grandmaster_clock_variance) {
      return true;
    }
    return false;
  }
};

struct __attribute__((packed)) SyncMessageV1 {
private:
  TimestampV1 precisionOriginTimestamp;
  uint16_t epoch = 0;
  uint16_t current_utc_offset = 0;
  uint8_t dummy = 0;
  GrandMasterInfo gm_info;
  uint8_t dummy6[3]{};
  int8_t sync_interval = 0;
  uint8_t dummy7[2]{};
  uint16_t localClock_variance = gm_info.grandmaster_clock_variance;
  uint8_t dummy8[2]{};
  uint16_t localSteps_removed = 0;
  uint8_t dummy9[3]{};
  uint8_t localClock_stratum = gm_info.grandmaster_clock_stratum;
  id_type_t local_clock_id;
  uint8_t dummy10 = 0;
  uint8_t parent_communication_technology = GRANDMASTER_ETHERNET_TECH;
  uuid_type_t parent_uuid{};
  uint8_t dummy11[2]{};
  uint16_t parent_port_field = 0;
  uint8_t dummy12[2]{};
  uint16_t estimated_master_variance = 0;
  uint32_t estimated_master_drift = 0;
  uint8_t dummy13[3]{};
  uint8_t utc_reasonable = 0;

public:
  SyncMessageV1(const TimestampV1 &ts, const uuid_type_t &my_mac_address,
                sequence_id_t seq_id, const network::NetworkingProfile &profile);

  int get_gm_clock_stratum() const {
    return gm_info.grandmaster_clock_stratum;
  }
  int get_local_clock_stratum() const {
    return localClock_stratum;
  }

  const GrandMasterInfo &get_gm_info() const {
    return gm_info;
  }

  void set_gm_id(const GrandMasterInfo &info) {
    this->gm_info = info;
  }

  const TimestampV1 &get_precisionOriginTimestamp() const {
    return precisionOriginTimestamp;
  }

  std::string get_grandmaster_uuid() const {
    return get_gm_info().get_uuid();
  }

  std::string get_parent_uuid() const {
    return uuid_to_string(parent_uuid);
  }
};

struct __attribute__((packed)) DelayRequestV1 : public SyncMessageV1 {
public:
  DelayRequestV1(const TimestampV1 &ts, const uuid_type_t &my_mac_address,
                 sequence_id_t seq_id, const network::NetworkingProfile &profile)
      : SyncMessageV1(ts, my_mac_address, seq_id, profile) {
  }
};

enum class PtpV1_PacketControlValue {
  PTP_SYNC_MESSAGE = 0,
  PTP_DELAY_REQ_MESSAGE,
  PTP_FOLLOWUP_MESSAGE,
  PTP_DELAY_RESP_MESSAGE,
  PTP_MANAGEMENT_MESSAGE,
  PTP_SYNC_MESSAGE_BURST,
  PTP_DELAY_REQ_MESSAGE_BURST
};

struct __attribute__((packed)) PtpHeaderV1 {
private:
  uint16_t versionPTP = 0;                                           // 2 0
  uint16_t versionNetwork = 0;                                       // 2 2
  char subdomain[16]{};                                              // 16 4
  uint8_t messageType = 0;                                           // 1 20
  uint8_t sourceCommunicationTechnology = GRANDMASTER_ETHERNET_TECH; // 1 21
  uuid_type_t sourceUUID{};                                          //   6 22
  uint16_t sourcePortId = 1;                                         // 2 28
  sequence_id_t sequenceId = 0;                                      // 2 30
  uint8_t control = 0;                                               // 1 32
  uint8_t reserved = 0;
  uint16_t flags = 0; // 2 34
  uint32_t reserved2 = 0;

public:
  // for set_flags
  static constexpr uint16_t MASK_PTP_LI61 = (1 << 0);
  static constexpr uint16_t MASK_PTP_LI59 = (1 << 1);
  static constexpr uint16_t MASK_PTP_BOUNDARY_CLOCK = (1 << 2);
  static constexpr uint16_t MASK_PTP_ASSIST = (1 << 3);
  static constexpr uint16_t MASK_PTP_EXT_SYNC = (1 << 4);
  static constexpr uint16_t MASK_PTP_PARENT_STATS = (1 << 5);
  static constexpr uint16_t MASK_PTP_SYNC_BURST = (1 << 6);

  static constexpr uint16_t MASK_ALT_MASTER = (1 << 8);
  static constexpr uint16_t MASK_TWO_STEP = (1 << 9);
  static constexpr uint16_t MASK_UNICAST = (1 << 10);

public:
  /*
      PtpHeaderV1(const PtpHeaderV1& c){
          memcpy(this, &c, sizeof(c));
      }
  */
  PtpHeaderV1(PtpV1_PacketControlValue _control, sequence_id_t seq,
              const uuid_type_t &uuid) {
    versionPTP = ntohs(1);
    versionNetwork = ntohs(1);
    strcpy(subdomain, "_DFLT");
    switch (_control) {
    case PtpV1_PacketControlValue::PTP_SYNC_MESSAGE:
      messageType = 1;
      break;

    case PtpV1_PacketControlValue::PTP_FOLLOWUP_MESSAGE:
    default:
      messageType = 2;
      break;
    }
    sourceCommunicationTechnology = GRANDMASTER_ETHERNET_TECH;
    sourceUUID = uuid;
    sourcePortId = 0;
    sequenceId = ntohs(seq);
    control = static_cast<uint8_t>(_control);
    reserved = 0;
    flags = 0;
    reserved2 = 0;
  }

  //  PtpHeaderV1(const PtpHeaderV1&&) = delete;

  uint16_t get_version_ptp() const {
    return ntohs(versionPTP);
  }

  uint16_t get_version_network() const {
    return ntohs(versionNetwork);
  }

  sequence_id_t get_sequence_id() const {
    return ntohs(sequenceId);
  }

  uint16_t get_source_port_id() const {
    return ntohs(sourcePortId);
  }

  void set_flags(uint16_t f) {
    flags = ntohs(f);
  }

  uint16_t get_flags() const {
    return ntohs(flags);
  }

  uint8_t get_control_direct() const {
    return control;
  }
  PtpV1_PacketControlValue get_control() const {
    return static_cast<PtpV1_PacketControlValue>(control);
  }

  bool is_ptp_LI61() const {
    return get_flags() & (1 << 0);
  }
  bool is_ptp_LI59() const {
    return get_flags() & (1 << 1);
  }
  bool is_ptp_boundary_clock() const {
    return get_flags() & (1 << 2);
  }
  bool is_ptp_assist() const {
    return get_flags() & (1 << 3);
  }
  bool is_ptp_ext_sync() const {
    return get_flags() & (1 << 4);
  }
  bool is_ptp_parent_stats() const {
    return get_flags() & (1 << 5);
  }
  bool is_ptp_sync_burst() const {
    return get_flags() & (1 << 6);
  }

  std::string get_source_uuid() const {
    return uuid_to_string(sourceUUID);
  }

  const uuid_type_t &get_source_uuid_raw() const {
    return sourceUUID;
  }

  PtpTech get_tech() const {
    switch (sourceCommunicationTechnology) {
    case 1:
      return PtpTech::ETHERNET;
    default:
      fprintf(stderr, "unhandled source comm tech\n");
      abort();
    }
  }

  u_int8_t get_message_type() const {
    return messageType;
  }

  std::string get_domain() const {
    return std::string(subdomain, sizeof(subdomain));
  }

  bool have_valid_domain() const {
    for (std::size_t i = 0; i < sizeof(subdomain); i++) {
      const char ch = subdomain[i];
      if (ch == 0) {
        if (i == 0) {
          return false;
        }
        break;
      }

      if (ch == '_') {
        // ok
      } else if (isalnum(ch)) {
        // ok
      } else {
        return false;
      }
    }
    return true;
  }
};

uuid_type_t convert_string_to_uuid(const std::string &mac, logging::ILogger &logger);
uuid_type_t convert_string_to_uuid(const std::optional<iuring::MacAddress> &mac, logging::ILogger &logger);

} // namespace ptp::v1
#pragma once

#include <netinet/in.h>
#include <sys/types.h>

#include <memory>

#include <slogger/StringUtils.hpp>

#include <iuring/IOUringInterface.hpp>

namespace ptp::v2 {
using sequence_id_t = uint16_t;

enum class MessageType : uint8_t {
  SYNC = 0,
  DELAY_REQ = 0x1,
  P_DELAY_REQ = 0x2,
  P_DELAY_RESP = 0x3,
  FOLLOW_UP = 0x8,
  DELAY_RESPONSE = 0x9,
  P_DELAY_RESPONSE_FOLLOW_UP = 0xa,
  ANNOUNCE = 0xb,
  SIGNALLING = 0xc,
  MANAGEMENT = 0xd
};

struct __attribute__((packed)) clock_identity_t {
  uint8_t data[8];

  std::string internal_to_string(const char *sep, bool short_form) const {
    std::string ret;
    for (int i = 0; i < 8; i++) {
      if (short_form) {
        if (i == 3 || i == 4) {
          continue;
        }
      }
      const auto v = data[i];
      if (i != 0) {
        ret += sep;
      }
      char buf[32];
      sprintf(buf, "%.02x", v);
      ret += buf;
    }
    return ret;
  }

  std::string to_string_short() const {
    return internal_to_string("", true);
  }

  std::string to_string_with_dashes() const {
    return StringUtils::to_upper(internal_to_string("-", false));
  }

  std::string to_string() const {
    return internal_to_string(":", false);
  }
};

enum class ControlField {
  SYNC_MESSAGE = 0,
  FOLLOW_UP_MESSAGE = 2,
  OTHER_MESSAGE = 5
};

struct __attribute__((packed)) PtpHeaderV2 {
private:
  u_int8_t transport_specific_and_message_type;
  uint8_t reserved_and_version_ptp = 2;
  uint16_t msg_length;
  uint8_t domain_number = 0;
  uint8_t minor_SdoId = 0;
  uint8_t flag_field0 = 0;
  uint8_t flag_field1 = 0;
  uint64_t correction_field = 0; // nanoseconds
  uint32_t message_type_specific = 0;
  clock_identity_t m_clock_identity;
  uint16_t source_port = 0;
  sequence_id_t sequence_number = 0;
  uint8_t control_field = 0;
  int8_t m_log_msg_interval = -2;

public:
  PtpHeaderV2(MessageType type, uint16_t len, sequence_id_t seq, ControlField cf,
              const clock_identity_t &id, int log_msg_interval)
      : m_clock_identity(id) {
    transport_specific_and_message_type =
        static_cast<std::underlying_type_t<MessageType>>(type);
    msg_length = ntohs(len);

    control_field = static_cast<std::underlying_type_t<ControlField>>(cf);

    sequence_number = ntohs(seq);

    m_log_msg_interval = log_msg_interval;

    if ((type == MessageType::SYNC) || (type == MessageType::P_DELAY_RESP)) {
      flag_field0 |= (1 << 1); // two step
    }
  }

  int8_t get_log_msg_interval() const { return m_log_msg_interval; }
  void set_log_msg_interval(int8_t v) { m_log_msg_interval = v; }

  uint64_t get_correction_raw() const {
    return correction_field;
  }

  void set_correction_raw(uint64_t cor) {
    correction_field = cor;
  }

  void set_domain_number(uint8_t domain_number) {
    this->domain_number = domain_number;
  }

  uint16_t get_source_port() const {
    return htons(source_port);
  }

  sequence_id_t get_seq_number() const {
    return htons(sequence_number);
  }

  const clock_identity_t &get_clock_id() const {
    return m_clock_identity;
  }

  uint16_t get_version_ptp_major() const {
    return reserved_and_version_ptp & 0b111;
  }

  uint16_t get_version_ptp_minor() const {
    return (reserved_and_version_ptp >> 4) & 0b111;
  }

  uint16_t get_major_sdod_id() const {
    return transport_specific_and_message_type >> 4;
  }

  MessageType get_message_type() const {
    return (MessageType)get_message_type_direct();
  }

  int get_message_type_direct() const {
    return (transport_specific_and_message_type & 0b1111);
  }

  uint32_t get_msg_len() const {
    return ntohs(msg_length);
  }

  uint8_t get_domain() const {
    return domain_number;
  }

  bool have_valid_domain() const {
    return domain_number < 10;
  }

  bool alternative_master() const {
    switch (get_message_type()) {
    case MessageType::ANNOUNCE:
    case MessageType::SYNC:

    case MessageType::FOLLOW_UP:
    case MessageType::DELAY_RESPONSE:
      return flag_field0 & (1 << 0);
    default:
      break;
    }
    return false;
  }

  bool two_step() const {
    switch (get_message_type()) {
    case MessageType::SYNC:
    case MessageType::P_DELAY_RESP:
      return flag_field0 & (1 << 1);
    default:
      break;
    }
    return false;
  }

  bool unicast() const {
    return flag_field0 & (1 << 2);
  }

  bool ptp_profile_specific1() const {
    return flag_field0 & (1 << 5);
  }

  bool ptp_profile_specific2() const {
    return flag_field0 & (1 << 6);
  }

  bool leap61() const {
    switch (get_message_type()) {
    case MessageType::ANNOUNCE:
      return flag_field0 & (1 << 0);
    default:
      break;
    }
    return false;
  }

  bool leap59() const {
    switch (get_message_type()) {
    case MessageType::ANNOUNCE:
      return flag_field0 & (1 << 1);
    default:
      break;
    }
    return false;
  }

  bool currentUtcOffsetValid() const {
    switch (get_message_type()) {
    case MessageType::ANNOUNCE:
      return flag_field0 & (1 << 2);
    default:
      break;
    }
    return false;
  }

  bool ptpTimescale() const {
    switch (get_message_type()) {
    case MessageType::ANNOUNCE:
      return flag_field0 & (1 << 3);
    default:
      break;
    }
    return false;
  }

  bool timeTraceable() const {
    switch (get_message_type()) {
    case MessageType::ANNOUNCE:
      return flag_field0 & (1 << 4);
    default:
      break;
    }
    return false;
  }

  bool frequencyTraceable() const {
    switch (get_message_type()) {
    case MessageType::ANNOUNCE:
      return flag_field0 & (1 << 5);
    default:
      break;
    }
    return false;
  }
};

struct __attribute__((packed)) TimestampV2 {
private:
  uint16_t m_seconds_hi;
  uint32_t m_seconds_lo;
  uint32_t m_nanos;

public:
  TimestampV2(const std::chrono::microseconds &total_micros)
      : TimestampV2(std::chrono::duration_cast<std::chrono::nanoseconds>(
            total_micros)) {
  }

  TimestampV2(const std::chrono::nanoseconds &total_nanos) {
    const auto nanos = total_nanos.count() % 1000000000ull;
    const auto seconds = total_nanos.count() / 1000000000ull;

    m_seconds_hi = seconds >> 32;
    m_seconds_lo = ntohl(seconds & 0xffffffff);
    m_nanos = ntohl(nanos);
  }

  std::chrono::nanoseconds get_micros() const {
    const auto secs_as_micros = std::chrono::seconds(get_secs());
    const auto nanos_as_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::nanoseconds(get_nanos()));
    return secs_as_micros + nanos_as_micros;
  }

  uint64_t get_secs() const {
    return ntohl(m_seconds_lo) | (((uint64_t)ntohs(m_seconds_hi)) << 32);
  }

  uint32_t get_nanos() const {
    return ntohl(m_nanos);
  }

  std::string to_string() const {
    return std::format("{}.{}",
                       get_secs(),
                       get_nanos());
  }
};

enum class TimeSource : uint8_t {
  INTERNAL_OSCILATOR = 0xa0
};

struct __attribute__((packed)) GrandMasterInfo {
  int16_t current_utc_offset = 0;
  uint8_t dummy = 0;
  uint8_t priority1 = 249; // lowest wins
  uint8_t gm_clock_class = 249;
  uint8_t gm_clock_accuracy = 0xfe; // unknown
  uint16_t gm_clock_variance = 61545;
  uint8_t priority2 = 120;
  clock_identity_t gm_clock_identity;
  uint16_t local_steps_removed = 0;
  uint8_t time_source = static_cast<std::underlying_type_t<TimeSource>>(TimeSource::INTERNAL_OSCILATOR);

  /** @return true if 'this' better than 'info
   */
  bool is_better_than(const GrandMasterInfo &info) const {
    if (this->local_steps_removed < info.local_steps_removed) {
      return true;
    }

    if (gm_clock_variance < info.gm_clock_variance) {
      return true;
    }
    return false;
  }

  TimeSource get_time_source() {
    return (TimeSource)time_source;
  }
};

struct __attribute__((packed)) PtpAnnounceMsg {
private:
  TimestampV2 m_origin_timestamp;
  GrandMasterInfo gm_info;

public:
  PtpAnnounceMsg(
      const TimestampV2 &origin_timestamp,
      const clock_identity_t &id,
      const network::NetworkingProfile &profile)
      : m_origin_timestamp(origin_timestamp) {
    static_assert(sizeof(m_origin_timestamp) == (6 + 4));
    static_assert(__builtin_offsetof(PtpAnnounceMsg,
                                     gm_info.current_utc_offset) == (6 + 4));

    gm_info.gm_clock_identity = id;
    gm_info.priority1 = profile.m_v2_priority1;
    gm_info.priority2 = profile.m_v2_priority2;
    gm_info.gm_clock_class = profile.m_v2_clock_class;
  }

  const GrandMasterInfo &get_info() const {
    return gm_info;
  }

  TimeSource get_time_source() {
    return gm_info.get_time_source();
  }

  const TimestampV2 &get_origin_timestamp() const { return m_origin_timestamp; }
};

struct __attribute__((packed)) PtpDelayRequest {
private:
  TimestampV2 m_origin_timestamp;

public:
  PtpDelayRequest(const TimestampV2 &origin_timestamp)
      : m_origin_timestamp(origin_timestamp) {
  }

  const TimestampV2 &get_origin_timestamp() const { return m_origin_timestamp; }
};

struct __attribute__((packed)) PtpDelayResponse {
private:
  TimestampV2 m_origin_timestamp;
  clock_identity_t m_clock_identity;
  uint16_t m_port_number;

public:
  PtpDelayResponse(const TimestampV2 &origin_timestamp, const clock_identity_t &clock_identity, uint16_t port_number)
      : m_origin_timestamp(origin_timestamp),
        m_clock_identity(clock_identity),
        m_port_number(htons(port_number)) {
  }

  uint16_t get_port_number() const { return htons(m_port_number); }

  const TimestampV2 &get_origin_timestamp() const { return m_origin_timestamp; }
};

struct __attribute__((packed)) PtpSyncMessage {
private:
  TimestampV2 m_origin_timestamp;

public:
  PtpSyncMessage(const TimestampV2 &origin_timestamp)
      : m_origin_timestamp(origin_timestamp) {
  }

  const TimestampV2 &get_origin_timestamp() const {
    return m_origin_timestamp;
  }
};

struct __attribute__((packed)) PtpFollowUpMessage {
private:
  TimestampV2 m_origin_timestamp;

public:
  PtpFollowUpMessage(const TimestampV2 &origin_timestamp)
      : m_origin_timestamp(origin_timestamp) {
  }
};

const clock_identity_t &get_local_clock_identity(
    iuring::NetworkAdapter &adapter, logging::ILogger &logger);

} // namespace ptp::v2

#include <ptp/PtpHeaderV1.hpp>
#include <ptp/PtpHeaderV2.hpp>
#include <ptp/PtpService-common.hpp>
#include <ptp/PtpV1Handler.hpp>
#include <ptp/PtpV2Handler.hpp>

namespace ptp {
class IPtpService {
public:
  virtual v2::clock_identity_t get_v2_clk_id() = 0;
  virtual std::chrono::nanoseconds get_time() const = 0;
};
} // namespace ptp

#include <chrono>
#include <optional>
#include <thread>

#include <iuring/IOUringInterface.hpp>

#include <slogger/ILogger.hpp>
#include <slogger/ITimer.hpp>
#include <slogger/TimeUtils.hpp>

#include <audio/NetworkProfiles.hpp>
#include <ptp/PtpHeaderV2.hpp>

#include <urtsched/RealtimeKernel.hpp>
#include <urtsched/Service.hpp>

namespace ptp {
enum class PtpState {
  UNKNOWN,
  LISTENING,
  LEADER,
  FOLLOWER,
};

static inline const char *state_to_string(PtpState s) {
  switch (s) {
  case PtpState::UNKNOWN:
    return "unknown";
  case PtpState::LISTENING:
    return "listening";
  case PtpState::LEADER:
    return "leader";
  case PtpState::FOLLOWER:
    return "follower";
  }
  return "unknown ptp state";
}

enum class PtpVersion {
  v1,
  v2,
  v2_1
};

class PtpService;

class CommonState : public service::Service {
public:
  CommonState(
      const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
      const std::shared_ptr<iuring::IOUringInterface> &io,
      iuring::NetworkAdapter &adapter,
      PtpService &service,
      const std::string &my_ptp_version);

  error::Error finish() override { return error::Error::OK; }

  const std::shared_ptr<iuring::ISocket> &get_ptp_event_socket();
  const std::shared_ptr<iuring::ISocket> &get_ptp_general_socket();
  const network::NetworkingProfile &get_profile() const;
  PtpState get_state() const {
    return m_state;
  }

  void step();

  virtual bool have_grand_master() = 0;
  virtual void handle_listening_state();
  virtual void handle_follower_state();
  virtual void handle_leader_state() = 0;

  const std::shared_ptr<iuring::IOUringInterface> &get_io() const {
    return m_io;
  }

public:
  PtpService &m_service;
  std::string m_my_ptp_version;

  PtpState m_state = PtpState::UNKNOWN;
  std::optional<time_utils::Timeout> m_listen_timeout;

  std::shared_ptr<realtime::PeriodicTask> m_ping_master_periodic;
  std::shared_ptr<realtime::PeriodicTask> m_sync_periodic;
  std::shared_ptr<realtime::PeriodicTask> m_announce_periodic;

  std::shared_ptr<iuring::IOUringInterface> m_io;
  iuring::NetworkAdapter &m_adapter;
};

} // namespace ptp

#include <chrono>
#include <optional>
#include <thread>

#include <Configuration.hpp>

#include <iuring/IOUringInterface.hpp>

#include <slogger/ILogger.hpp>
#include <slogger/TimeUtils.hpp>

#include <urtsched/IService.hpp>

#include <iuring/ISocketFactory.hpp>

#include "IPtpService.hpp"

namespace ptp {
class PtpService : public IPtpService, public service::IService {
public:
  PtpService(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
             const std::shared_ptr<iuring::IOUringInterface> &network,
             logging::ILogger &logger, const settings::Configuration &config,
             iuring::NetworkAdapter &adapter, iuring::ISocketFactory &socket_factory)
      : m_socket_factory(socket_factory), m_logger(logger), m_io(network), m_adapter(adapter), m_config(config) {
    if (m_config.ptp_v1_enabled) {
      m_state_v1 = std::make_shared<PtpV1Handler>(
          rt_kernel, network, adapter, *this);
    }

    if (m_config.ptp_v2_enabled) {
      m_state_v2 = std::make_shared<PtpV2Handler>(
          rt_kernel, network, adapter, *this);
    }
  }

  std::chrono::nanoseconds get_time() const override;

  std::string get_service_status_as_json() const override;

  ~PtpService();

  bool allow_clock_leader() const {
    return m_allow_clock_leader;
  }

  void disable_allow_clock_leader() {
    m_allow_clock_leader = false;
  }

  void enable_allow_clock_leader() {
    m_allow_clock_leader = true;
  }

  [[nodiscard]] error::Error init();
  [[nodiscard]] error::Error finish() override {
    return error::Error::OK;
  }

  void step_v1();
  void step_v2();

  static iuring::IPAddress PTP_PRIMARY_MCAST_IPADDR;

  static iuring::IPAddress PTP_PDELAY_MCAST_IPADDR;

  v2::clock_identity_t get_v2_clk_id() override;

  const network::NetworkingProfile &get_profile() const {
    return m_profile;
  }

  logging::ILogger &get_logger() const {
    return m_logger;
  }

  std::shared_ptr<iuring::IOUringInterface> &get_io() {
    return m_io;
  }

  const std::shared_ptr<iuring::ISocket> &get_ptp_event_socket() {
    return m_ptp_event_socket;
  }

  const std::shared_ptr<iuring::ISocket> &get_ptp_general_socket() {
    return m_ptp_general_socket;
  }

  bool enabled_ptp_v1() const {
    return m_state_v1 != nullptr;
  }

  std::string get_v1_leader_uuid() const {
    if (m_state_v1) {
      return m_state_v1->get_leader_uuid();
    }
    return "N/A";
  }

  std::string get_v2_leader_uuid() const {
    if (m_state_v2) {
      return m_state_v2->get_leader_uuid();
    }
    return "N/A";
  }

  std::string get_v1_local_uuid() const {
    if (m_state_v1) {
      return m_state_v1->get_local_uuid();
    }
    return "N/A";
  }

  std::string get_v2_local_uuid() const {
    if (m_state_v2) {
      return m_state_v2->get_local_uuid();
    }
    return "N/A";
  }

  bool enabled_ptp_v2() const {
    return m_state_v2 != nullptr;
  }

  uint32_t num_ptpv1_packets_sent() const {
    if (m_state_v1) {
      return m_state_v1->num_packets_sent();
    }
    return 0;
  }

  uint32_t num_ptpv2_packets_sent() const {
    if (m_state_v2) {
      return m_state_v2->num_packets_sent();
    }
    return 0;
  }

  uint32_t num_ptpv1_packets_received() const {
    if (m_state_v1) {
      return m_state_v1->num_packets_received();
    }
    return 0;
  }

  uint32_t num_ptpv2_packets_received() const {
    if (m_state_v2) {
      return m_state_v2->num_packets_received();
    }
    return 0;
  }

  PtpState get_state_v1() const {
    if (m_state_v1) {
      return m_state_v1->get_state();
    }
    return PtpState::UNKNOWN;
  }

  PtpState get_state_v2() const {
    if (m_state_v2) {
      return m_state_v2->get_state();
    }
    return PtpState::UNKNOWN;
  }

  iuring::NetworkAdapter &get_adapter() {
    return m_adapter;
  }

private:
  iuring::ISocketFactory &m_socket_factory;
  network::NetworkingProfile m_profile = network::AES76_profile;

  bool m_allow_clock_leader = true;

  std::shared_ptr<PtpV1Handler> m_state_v1;
  std::shared_ptr<PtpV2Handler> m_state_v2;

  logging::ILogger &m_logger;

  std::shared_ptr<iuring::IOUringInterface> m_io;

  std::shared_ptr<iuring::ISocket> m_ptp_event_socket;
  std::shared_ptr<iuring::ISocket> m_ptp_general_socket;

  iuring::NetworkAdapter &m_adapter;
  const settings::Configuration &m_config;

  int m_ethtool_sockfd = -1;
  int m_phc_fd = -1;

  void process_event(const iuring::ReceivedMessage &data);
  void process_general(const iuring::ReceivedMessage &data);

  [[nodiscard]] error::Error init_event_socket();

  [[nodiscard]] error::Error init_general_socket();

  void handle_incoming_ptp_packet(const iuring::ReceivedMessage &payload);

  std::expected<std::chrono::nanoseconds, std::string> get_ptp_time_from_ethernet_adapter() const;
};
} // namespace ptp

#include <chrono>
#include <optional>
#include <thread>

#include <iuring/IOUringInterface.hpp>

#include <slogger/ILogger.hpp>
#include <slogger/TimeUtils.hpp>

#include <ptp/PtpService-common.hpp>

namespace ptp {
struct GrandMasterV1 {
  v1::GrandMasterInfo m_info;
  iuring::IPAddress m_address;
};

struct PtpV1Handler : public CommonState {
public:
  explicit PtpV1Handler(
      const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
      const std::shared_ptr<iuring::IOUringInterface> &io,
      iuring::NetworkAdapter &adapter,
      PtpService &service)
      : CommonState(rt_kernel, io, adapter, service, "ptpv1") {
  }

  error::Error init() override;

  uint32_t num_packets_sent() const {
    return m_num_pkts_sent;
  }
  uint32_t num_packets_received() const {
    return m_num_pkts_received;
  }

  std::string get_leader_uuid();
  const std::string &get_local_uuid();

  void handle_ptpv1_packet(const iuring::ReceivedMessage &payload);

private:
  uint32_t m_num_pkts_sent = 0;
  uint32_t m_num_pkts_received = 0;
  v1::sequence_id_t m_v1_seq_id = 0;
  v1::sequence_id_t m_v1_general_seq_id = 0;

  std::optional<GrandMasterV1> m_grand_master;

  void send_ping_master_packet();

  bool have_grand_master() override {
    return m_grand_master.has_value();
  }

  void send_v1_sync_packet();
  void send_v1_follow_up_msg(v1::sequence_id_t follow_seq);
  void send_delay_reply(const v1::PtpHeaderV1 *delay_request_header,
                        const v1::TimestampV1 &req_ts,
                        const iuring::IPAddress &source_address);

  void handle_sync_message(const v1::PtpHeaderV1 *hdr, const iuring::ReceivedMessage &payload);
  void handle_delay_req_message(const v1::PtpHeaderV1 *hdr, const iuring::ReceivedMessage &payload);
  void handle_follow_up_message(const v1::PtpHeaderV1 *hdr);

  void handle_leader_state() override;

  bool is_better_than_current_master(const v1::SyncMessageV1 &msg);
  void set_clock_leader(
      const v1::GrandMasterInfo &gm_info, const iuring::IPAddress &address);
};

} // namespace ptp

#include <chrono>
#include <optional>
#include <thread>

#include <iuring/IOUringInterface.hpp>

#include <slogger/ILogger.hpp>
#include <slogger/TimeUtils.hpp>

#include <ptp/PtpService-common.hpp>

namespace ptp {
struct GrandMasterV2 {
  v2::GrandMasterInfo m_info;
  iuring::IPAddress m_address;
};

struct PtpV2Handler : public CommonState {
public:
  explicit PtpV2Handler(
      const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
      const std::shared_ptr<iuring::IOUringInterface> &io,
      iuring::NetworkAdapter &adapter,
      PtpService &service)
      : CommonState(rt_kernel, io, adapter, service, "ptpv2") {
  }

  error::Error init() override;

  std::optional<v2::clock_identity_t> get_v2_clock_leader_id() const {
    return m_v2_clock_leader_id_opt;
  }

  void handle_ptpv2_packet(const iuring::ReceivedMessage &payload);

  uint32_t num_packets_sent() const {
    return m_num_pkts_sent;
  }
  uint32_t num_packets_received() const {
    return m_num_pkts_received;
  }

  std::string get_leader_uuid();
  const std::string &get_local_uuid();

  bool is_better_than_current_master(const v2::GrandMasterInfo &info);

  bool is_clock_leader() {
    return get_leader_uuid() == get_local_uuid();
  }

private:
  uint32_t m_num_pkts_sent = 0;
  uint32_t m_num_pkts_received = 0;

  v2::sequence_id_t m_v2_seq_id = 0;
  v2::sequence_id_t m_v2_general_seq_id = 0;

  std::optional<GrandMasterV2> m_grand_master;

  // Track the time we sent DELAY_REQ for delay calculation
  std::chrono::nanoseconds m_delay_req_send_time{0};
  // Mean path delay to master in nanoseconds
  std::chrono::nanoseconds m_mean_path_delay{0};

  void send_ping_master_packet();

  void send_delay_reply(const v2::PtpHeaderV2 *header,
                        const v2::TimestampV2 &ts, const iuring::IPAddress &source_address);
  void send_v2_announce_packet();
  void send_v2_sync_packet();
  void send_v2_follow_up_msg(v2::sequence_id_t follow_seq);

  // either our ID or the selected leader's ID.
  std::optional<v2::clock_identity_t> m_v2_clock_leader_id_opt;

  void handle_leader_state() override;
  bool have_grand_master() override {
    return m_grand_master.has_value();
  }

  void set_clock_leader(
      const v2::GrandMasterInfo &gm_info, const iuring::IPAddress &address);
};

} // namespace ptp

#include <chrono>
#include <functional>
#include <iuring/IPAddress.hpp>
#include <map>

namespace utils {
using histogram_bucket_t = std::map<std::chrono::microseconds, uint64_t>;

using histogram_visitor_t = std::function<void(
    const iuring::IPAddress &addr, const histogram_bucket_t &buckets)>;

class BucketedHistogram {
public:
  void visit(const histogram_visitor_t &v) const {
    v(m_address, m_buckets);
  }

  void record_time(
      const iuring::IPAddress &addr, const std::chrono::nanoseconds &ts) {
    record_time(ts);
    m_address = addr;
  }

  void record_time(const std::chrono::nanoseconds &ts);

private:
  iuring::IPAddress m_address;
  std::optional<std::chrono::nanoseconds> m_last_arrival;
  histogram_bucket_t m_buckets;
};

class Histogram {
public:
  Histogram() {}

  void clear() {
    m_recv_info.clear();
  }

  void record_packet_arrival(
      const iuring::IPAddress &address, const std::chrono::nanoseconds &ts) {
    m_num_packets_received++;

    auto &map = m_recv_info[address];
    map.record_time(address, ts);
  }

  void record_rtp_packet_sent(const std::chrono::nanoseconds &ts) {
    m_num_rtp_packets_sent++;
    m_rtp_sent_info.record_time(ts);
  }

  uint64_t num_rtp_packets_sent() const {
    return m_num_rtp_packets_sent;
  }

  uint64_t num_packets_received() const {
    return m_num_packets_received;
  }

  void visit_recv(const histogram_visitor_t &v) const;

  void inc_num_audio_not_ready() { m_num_dropped_rtp_packets_audio_not_ready++; }
  void inc_num_dropped_rtp_queued_already() { m_num_dropped_rtp_sent_queued_already++; }
  void inc_num_dropped_rtp_chunk_locked() { m_num_dropped_rtp_chunk_locked++; }
  uint64_t get_num_dropped_rtp_sent_queued_already() const { return m_num_dropped_rtp_sent_queued_already; }
  uint64_t get_num_dropped_rtp_audio_not_ready() const { return m_num_dropped_rtp_packets_audio_not_ready; }

  std::string sent_to_json_string() const;

  std::string recv_to_json_string() const;

private:
  uint64_t m_num_rtp_packets_sent = 0;
  uint64_t m_num_packets_received = 0;
  uint64_t m_num_dropped_rtp_sent_queued_already = 0;
  uint64_t m_num_dropped_rtp_packets_audio_not_ready = 0;
  uint64_t m_num_dropped_rtp_chunk_locked = 0;

  BucketedHistogram m_rtp_sent_info;
  std::map<iuring::IPAddress, BucketedHistogram> m_recv_info;

  void handle_json(std::string &ret, const iuring::IPAddress &addr,
                   const histogram_bucket_t &buckets) const;
};

} // namespace utils

#include <queue>
#include <string>

#include <slogger/ILogger.hpp>

#include <iuring/IOUringInterface.hpp>

#include <urtsched/RealtimeKernel.hpp>
#include <urtsched/Service.hpp>
#include <urtsched/ServiceBus.hpp>

#include "http/HttpServer.hpp"

namespace Statistics {
class Statistics_Service : public service::Service {
public:
  Statistics_Service(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
                     const std::shared_ptr<iuring::IOUringInterface> &network, logging::ILogger &logger,
                     http::HttpServer &http_server, service::ServiceBus &bus)
      : service::Service(rt_kernel, logger), m_http_server(http_server), m_bus(bus), m_network(network) {
  }

  [[nodiscard]] error::Error init();
  [[nodiscard]] error::Error finish() override { return error::Error::OK; }

private:
  http::HttpServer &m_http_server;
  service::ServiceBus &m_bus;
  std::shared_ptr<iuring::IOUringInterface> m_network;

  std::string get_info_json();

  std::shared_ptr<iuring::IOUringInterface> &get_io() {
    return m_network;
  }
};
} // namespace Statistics
#pragma once

#include <memory>

#include <slogger/ILogger.hpp>

#include <iuring/IOUringInterface.hpp>

#include <urtsched/IService.hpp>
#include <urtsched/ServiceBus.hpp>

#include "Configuration.hpp"
#include <aoip/RavennaService.hpp>
#include <audio/AudioFormat.hpp>
#include <audio/AudioPipeline.hpp>
#include <domainmodel/ChannelMapper.hpp>
#include <domainmodel/Node.hpp>
#include <http/HttpServer.hpp>
#include <mdns/MDNS_Service.hpp>
#include <nmos/NMOS_Service.hpp>
#include <ptp/PtpService.hpp>
#include <statistics/Statistics_Service.hpp>

namespace audio {

class Application {
public:
  Application(const std::shared_ptr<iuring::IOUringInterface> &network,
              logging::ILogger &logger,
              const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
              service::ServiceBus &bus, settings::Configuration &config,
              iuring::NetworkAdapter &adapter,
              iuring::ISocketFactory &socket_factory,
              const std::shared_ptr<http::AbstractTLS> &tls,
              model::ChannelMapper &mapper,
              audio::IAlsaManager &alsa_manager);

  [[nodiscard]] error::Error init();
  [[nodiscard]] error::Error finish();

  void step() {
    m_rt_kernel->step();
  }

  // for integration test:
  std::shared_ptr<ptp::PtpService> get_ptp_service() {
    return m_ptp_service;
  }

private:
  iuring::ISocketFactory &m_socket_factory;
  service::ServiceBus &m_bus;
  std::shared_ptr<realtime::RealtimeKernel> m_rt_kernel;
  logging::ILogger &m_logger;
  std::shared_ptr<iuring::IOUringInterface> m_io;
  model::ChannelMapper &m_mapper;

  http::HttpClientManager m_http_client_manager;
  std::shared_ptr<ptp::PtpService> m_ptp_service;
  std::shared_ptr<audio::IRTP_Service> m_rtp_service;
  std::shared_ptr<model::Node> m_model;
  std::shared_ptr<audio::AudioPipeline> m_audio_pipeline;

  std::shared_ptr<mdns::MDNS_Service> m_mdns_server;
  std::shared_ptr<http::HttpServer> m_http_server;
  std::shared_ptr<NMOS::NMOS_Service> m_nmos_service;
  std::shared_ptr<Statistics::Statistics_Service> m_statistics_service;

  std::optional<std::shared_ptr<realtime::IdleTask>> m_idle_poll;
  std::optional<std::shared_ptr<realtime::IdleTask>> p_ptp_state_change;

  iuring::NetworkAdapter &m_adapter;
  const settings::Configuration &m_config;

  logging::ILogger &get_logger() {
    return m_logger;
  }

  iuring::NetworkAdapter &get_adapter() {
    return m_adapter;
  }
};

} // namespace audio

#include <slogger/ILogger.hpp>
#include <string>

#include <domainmodel/ChannelMapper.hpp>
#include <iuring/SocketPortID.hpp>
#include <urtsched/MultiCoreRealtimeKernel.hpp>

#include <http/HttpServer.hpp>

namespace settings {
class Configuration {
public:
  /** if returns 0, all good, otherwise exit code */
  int load(int argc, char **argv);

  void show_report();

public:
  http::ServerConfig http;

  // misc:
  bool show_debug = false;
  bool show_info = true;
  bool show_http_packets = false;

  // ethernet interface settings:
  std::string interface_name;
  bool tune_network_interface = false;

  // ptp settings:
  bool ptp_v1_enabled = true;
  bool ptp_v2_enabled = true;
  bool allow_clock_leader = true;

  // audio I/O settings:
  bool use_sine_gen = false;
  bool use_audio_hardware = false;
  std::vector<std::string> alsa_devices;

  // how long to run in test-mode
  int run_secs = -1;

  // audio settings:
  bool enable_audio = true;
  bool is_audio_server = true;
  bool is_audio_client = true;
  int num_audio_channels = 2;
  int num_sample_encoding_bits = 24;
  int sample_rate = 48000;

  audio::AudioProfile audio_profile;

  // realtime:
  realtime::CoreReservationMechanism core_reservation_mechanism =
      realtime::CoreReservationMechanism::NONE;
  uint32_t first_reserved_core = 2;

  // RTP settings (the interval between RTP packets sent)
  std::chrono::microseconds audio_latency = std::chrono::milliseconds(1);
  iuring::SocketPortID rtp_port = iuring::SocketPortID::RTP_AUDIO_PORT;
  iuring::IPAddress rtp_multicast_address =
      iuring::IPAddress::parse("239.69.11.44").value();
  bool listen_on_rtp_immediately = false;

  // NMOS settings:
  std::optional<std::string> nmos_config_file_name_opt; // = "nmos_config.json";

  // this many secs we send new keep-alives to the nmos registration server
  std::chrono::milliseconds nmos_keep_alive_interval =
      std::chrono::seconds(5);

  // after this many secs we delete the keep-alive
  std::chrono::milliseconds nmos_keep_alive_timeout =
      std::chrono::seconds(25);

  // SAP settings:
  bool enable_SAP = true;

  // SRTP encryption settings:
  bool enable_srtp_encryption = false;
};
} // namespace settings
 * \file certs.h
 *
 * \brief Sample certificates and DHM parameters for testing
 */
 /*
  *  Copyright The Mbed TLS Contributors
  *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
  */
#ifndef MBEDTLS_CERTS_H
#define MBEDTLS_CERTS_H

#include "mbedtls/build_info.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

   /* List of all PEM-encoded CA certificates, terminated by NULL;
    * PEM encoded if MBEDTLS_PEM_PARSE_C is enabled, DER encoded
    * otherwise. */
   extern const char *mbedtls_test_cas[];
   extern const size_t mbedtls_test_cas_len[];

   /* List of all DER-encoded CA certificates, terminated by NULL */
   extern const unsigned char *mbedtls_test_cas_der[];
   extern const size_t mbedtls_test_cas_der_len[];

#if defined(MBEDTLS_PEM_PARSE_C)
   /* Concatenation of all CA certificates in PEM format if available */
   extern const char mbedtls_test_cas_pem[];
   extern const size_t mbedtls_test_cas_pem_len;
#endif /* MBEDTLS_PEM_PARSE_C */

   /*
    * CA test certificates
    */

   extern const char mbedtls_test_ca_crt_ec_pem[];
   extern const char mbedtls_test_ca_key_ec_pem[];
   extern const char mbedtls_test_ca_pwd_ec_pem[];
   extern const char mbedtls_test_ca_key_rsa_pem[];
   extern const char mbedtls_test_ca_pwd_rsa_pem[];
   extern const char mbedtls_test_ca_crt_rsa_sha1_pem[];
   extern const char mbedtls_test_ca_crt_rsa_sha256_pem[];

   extern const unsigned char mbedtls_test_ca_crt_ec_der[];
   extern const unsigned char mbedtls_test_ca_key_ec_der[];
   extern const unsigned char mbedtls_test_ca_key_rsa_der[];
   extern const unsigned char mbedtls_test_ca_crt_rsa_sha1_der[];
   extern const unsigned char mbedtls_test_ca_crt_rsa_sha256_der[];

   extern const size_t mbedtls_test_ca_crt_ec_pem_len;
   extern const size_t mbedtls_test_ca_key_ec_pem_len;
   extern const size_t mbedtls_test_ca_pwd_ec_pem_len;
   extern const size_t mbedtls_test_ca_key_rsa_pem_len;
   extern const size_t mbedtls_test_ca_pwd_rsa_pem_len;
   extern const size_t mbedtls_test_ca_crt_rsa_sha1_pem_len;
   extern const size_t mbedtls_test_ca_crt_rsa_sha256_pem_len;

   extern const size_t mbedtls_test_ca_crt_ec_der_len;
   extern const size_t mbedtls_test_ca_key_ec_der_len;
   extern const size_t mbedtls_test_ca_pwd_ec_der_len;
   extern const size_t mbedtls_test_ca_key_rsa_der_len;
   extern const size_t mbedtls_test_ca_pwd_rsa_der_len;
   extern const size_t mbedtls_test_ca_crt_rsa_sha1_der_len;
   extern const size_t mbedtls_test_ca_crt_rsa_sha256_der_len;

   /* Config-dependent dispatch between PEM and DER encoding
    * (PEM if enabled, otherwise DER) */

   extern const char mbedtls_test_ca_crt_ec[];
   extern const char mbedtls_test_ca_key_ec[];
   extern const char mbedtls_test_ca_pwd_ec[];
   extern const char mbedtls_test_ca_key_rsa[];
   extern const char mbedtls_test_ca_pwd_rsa[];
   extern const char mbedtls_test_ca_crt_rsa_sha1[];
   extern const char mbedtls_test_ca_crt_rsa_sha256[];

   extern const size_t mbedtls_test_ca_crt_ec_len;
   extern const size_t mbedtls_test_ca_key_ec_len;
   extern const size_t mbedtls_test_ca_pwd_ec_len;
   extern const size_t mbedtls_test_ca_key_rsa_len;
   extern const size_t mbedtls_test_ca_pwd_rsa_len;
   extern const size_t mbedtls_test_ca_crt_rsa_sha1_len;
   extern const size_t mbedtls_test_ca_crt_rsa_sha256_len;

   /* Config-dependent dispatch between SHA-1 and SHA-256
    * (SHA-256 if enabled, otherwise SHA-1) */

   extern const char mbedtls_test_ca_crt_rsa[];
   extern const size_t mbedtls_test_ca_crt_rsa_len;

   /* Config-dependent dispatch between EC and RSA
    * (RSA if enabled, otherwise EC) */

   extern const char *mbedtls_test_ca_crt;
   extern const char *mbedtls_test_ca_key;
   extern const char *mbedtls_test_ca_pwd;
   extern const size_t mbedtls_test_ca_crt_len;
   extern const size_t mbedtls_test_ca_key_len;
   extern const size_t mbedtls_test_ca_pwd_len;

   /*
    * Server test certificates
    */

   extern const char mbedtls_test_srv_crt_ec_pem[];
   extern const char mbedtls_test_srv_key_ec_pem[];
   extern const char mbedtls_test_srv_pwd_ec_pem[];
   extern const char mbedtls_test_srv_key_rsa_pem[];
   extern const char mbedtls_test_srv_pwd_rsa_pem[];
   extern const char mbedtls_test_srv_crt_rsa_sha1_pem[];
   extern const char mbedtls_test_srv_crt_rsa_sha256_pem[];

   extern const unsigned char mbedtls_test_srv_crt_ec_der[];
   extern const unsigned char mbedtls_test_srv_key_ec_der[];
   extern const unsigned char mbedtls_test_srv_key_rsa_der[];
   extern const unsigned char mbedtls_test_srv_crt_rsa_sha1_der[];
   extern const unsigned char mbedtls_test_srv_crt_rsa_sha256_der[];

   extern const size_t mbedtls_test_srv_crt_ec_pem_len;
   extern const size_t mbedtls_test_srv_key_ec_pem_len;
   extern const size_t mbedtls_test_srv_pwd_ec_pem_len;
   extern const size_t mbedtls_test_srv_key_rsa_pem_len;
   extern const size_t mbedtls_test_srv_pwd_rsa_pem_len;
   extern const size_t mbedtls_test_srv_crt_rsa_sha1_pem_len;
   extern const size_t mbedtls_test_srv_crt_rsa_sha256_pem_len;

   extern const size_t mbedtls_test_srv_crt_ec_der_len;
   extern const size_t mbedtls_test_srv_key_ec_der_len;
   extern const size_t mbedtls_test_srv_pwd_ec_der_len;
   extern const size_t mbedtls_test_srv_key_rsa_der_len;
   extern const size_t mbedtls_test_srv_pwd_rsa_der_len;
   extern const size_t mbedtls_test_srv_crt_rsa_sha1_der_len;
   extern const size_t mbedtls_test_srv_crt_rsa_sha256_der_len;

   /* Config-dependent dispatch between PEM and DER encoding
    * (PEM if enabled, otherwise DER) */

   extern const char mbedtls_test_srv_crt_ec[];
   extern const char mbedtls_test_srv_key_ec[];
   extern const char mbedtls_test_srv_pwd_ec[];
   extern const char mbedtls_test_srv_key_rsa[];
   extern const char mbedtls_test_srv_pwd_rsa[];
   extern const char mbedtls_test_srv_crt_rsa_sha1[];
   extern const char mbedtls_test_srv_crt_rsa_sha256[];

   extern const size_t mbedtls_test_srv_crt_ec_len;
   extern const size_t mbedtls_test_srv_key_ec_len;
   extern const size_t mbedtls_test_srv_pwd_ec_len;
   extern const size_t mbedtls_test_srv_key_rsa_len;
   extern const size_t mbedtls_test_srv_pwd_rsa_len;
   extern const size_t mbedtls_test_srv_crt_rsa_sha1_len;
   extern const size_t mbedtls_test_srv_crt_rsa_sha256_len;

   /* Config-dependent dispatch between SHA-1 and SHA-256
    * (SHA-256 if enabled, otherwise SHA-1) */

   extern const char mbedtls_test_srv_crt_rsa[];
   extern const size_t mbedtls_test_srv_crt_rsa_len;

   /* Config-dependent dispatch between EC and RSA
    * (RSA if enabled, otherwise EC) */

   extern const char *mbedtls_test_srv_crt;
   extern const char *mbedtls_test_srv_key;
   extern const char *mbedtls_test_srv_pwd;
   extern const size_t mbedtls_test_srv_crt_len;
   extern const size_t mbedtls_test_srv_key_len;
   extern const size_t mbedtls_test_srv_pwd_len;

   /*
    * Client test certificates
    */

   extern const char mbedtls_test_cli_crt_ec_pem[];
   extern const char mbedtls_test_cli_key_ec_pem[];
   extern const char mbedtls_test_cli_pwd_ec_pem[];
   extern const char mbedtls_test_cli_key_rsa_pem[];
   extern const char mbedtls_test_cli_pwd_rsa_pem[];
   extern const char mbedtls_test_cli_crt_rsa_pem[];

   extern const unsigned char mbedtls_test_cli_crt_ec_der[];
   extern const unsigned char mbedtls_test_cli_key_ec_der[];
   extern const unsigned char mbedtls_test_cli_key_rsa_der[];
   extern const unsigned char mbedtls_test_cli_crt_rsa_der[];

   extern const size_t mbedtls_test_cli_crt_ec_pem_len;
   extern const size_t mbedtls_test_cli_key_ec_pem_len;
   extern const size_t mbedtls_test_cli_pwd_ec_pem_len;
   extern const size_t mbedtls_test_cli_key_rsa_pem_len;
   extern const size_t mbedtls_test_cli_pwd_rsa_pem_len;
   extern const size_t mbedtls_test_cli_crt_rsa_pem_len;

   extern const size_t mbedtls_test_cli_crt_ec_der_len;
   extern const size_t mbedtls_test_cli_key_ec_der_len;
   extern const size_t mbedtls_test_cli_key_rsa_der_len;
   extern const size_t mbedtls_test_cli_crt_rsa_der_len;

   /* Config-dependent dispatch between PEM and DER encoding
    * (PEM if enabled, otherwise DER) */

   extern const char mbedtls_test_cli_crt_ec[];
   extern const char mbedtls_test_cli_key_ec[];
   extern const char mbedtls_test_cli_pwd_ec[];
   extern const char mbedtls_test_cli_key_rsa[];
   extern const char mbedtls_test_cli_pwd_rsa[];
   extern const char mbedtls_test_cli_crt_rsa[];

   extern const size_t mbedtls_test_cli_crt_ec_len;
   extern const size_t mbedtls_test_cli_key_ec_len;
   extern const size_t mbedtls_test_cli_pwd_ec_len;
   extern const size_t mbedtls_test_cli_key_rsa_len;
   extern const size_t mbedtls_test_cli_pwd_rsa_len;
   extern const size_t mbedtls_test_cli_crt_rsa_len;

   /* Config-dependent dispatch between EC and RSA
    * (RSA if enabled, otherwise EC) */

   extern const char *mbedtls_test_cli_crt;
   extern const char *mbedtls_test_cli_key;
   extern const char *mbedtls_test_cli_pwd;
   extern const size_t mbedtls_test_cli_crt_len;
   extern const size_t mbedtls_test_cli_key_len;
   extern const size_t mbedtls_test_cli_pwd_len;

#ifdef __cplusplus
 }
#endif

#endif /* certs.h */
#pragma once

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

 struct SendResult {
   int status;
 };

 struct ConnectResult {
   int status;
   IPAddress m_address;
 };

 struct CloseResult {
   int status;
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

 private:
   std::optional<sockaddr_in> m_in4;
   std::optional<sockaddr_in6> m_in6;
 };

 /** util func for converting a 'a.b.c.d' IP address and
  * port to an IPAddress object
  */
 IPAddress create_sock_addr_in(
     const char *addr, const SocketPortID port, logging::ILogger &logger);

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
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cstring>

#include <slogger/ILogger.hpp>
#include <slogger/StringUtils.hpp>

#include <iuring/IPAddress.hpp>

 namespace iuring {
 class AcceptResult;

 class ISocket {
 public:
   ISocket(SocketType type, SocketPortID port, logging::ILogger &logger,
           SocketKind kind, int fd)
       : m_type(type), m_port(port), m_logger(logger), m_kind(kind), m_fd(fd) {
   }

   virtual ~ISocket() = default;

   int get_fd() const {
     return m_fd;
   }

   virtual int mcast_bind() = 0;
   virtual void join_multicast_group(
       const std::string &ip_address, const std::string &source_iface) = 0;

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

 private:
   SocketType m_type;
   SocketPortID m_port;
   logging::ILogger &m_logger;
   SocketKind m_kind;
   int m_fd;

 private:
   friend class SocketFactoryImpl;

   static std::shared_ptr<ISocket> create_impl(SocketType type,
                                               SocketPortID port, logging::ILogger &logger, SocketKind kind);

   static std::shared_ptr<ISocket> create_impl(
       logging::ILogger &logger, const AcceptResult &res);
 };

 } // namespace iuring

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

#include "MacAddress.hpp"

 namespace iuring {
 class NetworkAdapter {
 public:
   NetworkAdapter(logging::ILogger &logger, const std::string &interface_name, bool tune)
       : m_logger(logger),
         m_interface_name(interface_name), m_tune(tune) {}

   void init();

   void tune();

   const std::string &get_hostname() const { return m_hostname; }

   void set_interface_ip4(const std::string &ip) {
     m_interface_ip4 = ip;
     LOG_INFO(get_logger(), "interface IP4 set to {}", ip);
   }

   void set_interface_ip6(const std::string &ip) {
     m_interface_ip6 = ip;
     LOG_INFO(get_logger(), "interface IP6 set to {}", ip);
   }

   const std::string &get_interface_name() const {
     return m_interface_name;
   }

   std::optional<MacAddress> get_my_mac_address();

   const std::optional<std::string> get_interface_ip4() const {
     return m_interface_ip4;
   }

   const std::optional<std::string> get_interface_ip6() const {
     return m_interface_ip6;
   }

 private:
   std::string m_hostname;
   logging::ILogger &m_logger;

   std::optional<std::string> m_interface_ip4;
   std::optional<std::string> m_interface_ip6;
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
#pragma once

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
   UNENCRYPTED_WEB_PORT = 80,

   PTP_PORT_EVENT = 319,
   PTP_PORT_GENERAL = 320,

   ENCRYPTED_WEB_PORT = 443,

   LAST_PRIVILEDGED_PORT_ID = 1024,

   LOCAL_WEB_PORT = 8080,

   // Session Announcement Protocol
   SAP_PORT_EVENT = 9875,

   // rtp audio bcast
   RTP_AUDIO_PORT = 5004,

   MDNS_PORT = 5353,

   SAP_PORT = 9875,

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

#include <format>
#include <string>

 /**
  * @file Error.hpp
  * @brief Defines error codes and conversion functions for network operations.
  */

 namespace error {

 enum class Error {
   OK,
   RANGE,
   MMAP_FAILED,
   FAILED_TO_CREATE_SOCKET,
   FAILED_TO_OPEN_PCM,
   ALSA_FAILURE,
   NO_ALSA_CAPTURE,
   NO_ALSA_PLAYBACK,
   HOSTNAME_RESOLVE_FAILED,
   BAD_PPROTOCOL,
   INVALID_SDP,
   UNKNOWN
 };

 Error errno_to_error(int err);
 } // namespace error

 template <>
 struct std::formatter<error::Error> {
   constexpr auto parse(std::format_parse_context &ctx) {
     return ctx.begin();
   }

   auto format(const error::Error &c, std::format_context &ctx) const {
     switch (c) {
     case error::Error::OK:
       return std::format_to(ctx.out(), "OK");
     case error::Error::RANGE:
       return std::format_to(ctx.out(), "RANGE");
     case error::Error::MMAP_FAILED:
       return std::format_to(ctx.out(), "MMAP_FAILED");
     case error::Error::FAILED_TO_CREATE_SOCKET:
       return std::format_to(ctx.out(), "FAILED_TO_CREATE_SOCKET");
     case error::Error::FAILED_TO_OPEN_PCM:
       return std::format_to(ctx.out(), "FAILED_TO_OPEN_PCM");
     case error::Error::ALSA_FAILURE:
       return std::format_to(ctx.out(), "ALSA_FAILURE");
     case error::Error::NO_ALSA_CAPTURE:
       return std::format_to(ctx.out(), "NO_ALSA_CAPTURE");
     case error::Error::NO_ALSA_PLAYBACK:
       return std::format_to(ctx.out(), "NO_ALSA_PLAYBACK");
     case error::Error::HOSTNAME_RESOLVE_FAILED:
       return std::format_to(ctx.out(), "HOSTNAME_RESOLVE_FAILED");
     case error::Error::UNKNOWN:
       return std::format_to(ctx.out(), "UNKNOWN");
     case error::Error::BAD_PPROTOCOL:
       return std::format_to(ctx.out(), "BAD_PPROTOCOL");
     case error::Error::INVALID_SDP:
       return std::format_to(ctx.out(), "INVALID_SDP");
     }
     return std::format_to(ctx.out(), "UNKNOWN_ERROR_CODE");
   }
 };
#pragma once

 /**
  * @file ILogger.hpp
  * @brief Defines the ILogger interface for logging messages with different
  * severity levels.
  *
  * This interface provides methods for logging debug, info, and error messages.
  */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include <format>

 namespace logging {
 class ILogger {
 public:
   ILogger(bool debug, bool info)
       : m_debug(debug), m_info(info) {
   }

   virtual ~ILogger() {}

   virtual void debug_msg(
       uint32_t line, const char *file, const std::string msg) = 0;
   virtual void info_msg(
       uint32_t line, const char *file, const std::string msg) = 0;
   virtual void error_msg(
       uint32_t line, const char *file, const std::string msg) = 0;

   bool enable_debug() const {
     return m_debug;
   }
   bool enable_info() const {
     return m_info;
   }

 protected:
   bool m_debug = true;
   bool m_info = true;
 };

 const char *strip_prefix(const char *path);

#define LOG_DEBUG(logger, ...)                                  \
   if (logger.enable_debug()) {                                  \
     logger.debug_msg(__LINE__, logging::strip_prefix(__FILE__), \
                      std::format(__VA_ARGS__));                 \
   }

#define LOG_INFO(logger, ...)                                  \
   if (logger.enable_info()) {                                  \
     logger.info_msg(__LINE__, logging::strip_prefix(__FILE__), \
                     std::format(__VA_ARGS__));                 \
   }

#define LOG_INFO_ONCE(logger, ...)                               \
   do {                                                           \
     static bool init_once;                                       \
     if (logger.enable_info() && !init_once) {                    \
       logger.info_msg(__LINE__, logging::strip_prefix(__FILE__), \
                       std::format(__VA_ARGS__));                 \
       init_once = true;                                          \
     }                                                            \
   } while (0)

#define LOG_ERROR(logger, ...) \
   logger.error_msg(            \
       __LINE__, logging::strip_prefix(__FILE__), std::format(__VA_ARGS__))

#define LOG_ERROR_ONCE(logger, ...)                               \
   do {                                                            \
     static bool init_once;                                        \
     if (!init_once) {                                             \
       logger.error_msg(__LINE__, logging::strip_prefix(__FILE__), \
                        std::format(__VA_ARGS__));                 \
       init_once = true;                                           \
     }                                                             \
   } while (0)

#define LOG_ERROR_SLOW(logger, ...)                               \
   do {                                                            \
     static uint32_t counter;                                      \
     if (counter++ > 1000) {                                       \
       logger.error_msg(__LINE__, logging::strip_prefix(__FILE__), \
                        std::format(__VA_ARGS__));                 \
       counter = 0;                                                \
     }                                                             \
   } while (0)

 } // namespace logging

 /**
  * @file Logger.hpp
  * @brief Defines the Logger class for logging messages with different severity
  * levels.
  *
  * This class provides methods for logging debug, info, and error messages.
  * It supports logging to the console or to a file stream.
  */

#include <stdint.h>
#include <stdio.h>
#include <string>

#include "ILogger.hpp"

 namespace logging {

 enum class LogOutput {
   CONSOLE,
   FILE_STREAM
 };

 class Logger : public ILogger {
 public:
   void debug_msg(uint32_t line, const char *file, const std::string msg) override {
     log(line, file, Level::DEBUG, msg);
   }

   void info_msg(
       uint32_t line, const char *file, const std::string msg) override {
     log(line, file, Level::INFO, msg);
   }

   void error_msg(
       uint32_t line, const char *file, const std::string msg) override {
     log(line, file, Level::ERROR, msg);
   }

   Logger(bool debug, bool info, LogOutput output);
   ~Logger();

 private:
   LogOutput m_output;
   FILE *m_f = nullptr;

   enum class Level {
     DEBUG,
     INFO,
     ERROR
   };

   void log(
       uint32_t line, const char *file, Level level, const std::string &msg);
 };

 } // namespace logging

#include <string>

#include <slogger/ILogger.hpp>

 namespace shell {
 enum class RunOpt {
   LOG_ERROR_AS_WARNING,
   ABORT_ON_ERROR
 };

 void run_cmd(const std::string &cmd, logging::ILogger &logger, RunOpt opt);
 } // namespace shell

#include <array>
#include <map>
#include <optional>
#include <stdint.h>
#include <string>
#include <vector>

 /**
  * note: these functions are functional: they return the result and do not
  * change their arguments
  */

 namespace StringUtils {
 /** We get name lists like: [foo, local], then check if the last equals "local"
  */
 bool last_item_equals(const std::vector<std::string> &name_list, const std::string &last_item_name);

 template <typename T>
 bool contains(const std::vector<T> &l, const T elt) {
   for (auto it : l) {
     if (it == elt) {
       return true;
     }
   }
   return false;
 }

 [[maybe_unused]]
 static inline std::string to_string(bool v) {
   return v ? "true" : "false";
 }

 [[maybe_unused]]
 static inline std::string to_string(int v) {
   return std::to_string(v);
 }

 [[maybe_unused]]
 static inline std::string to_string(const std::string &v) {
   return v;
 }

 /** @returns comma seperated string
  */
 template <typename T, size_t N>
 std::string array_to_string(const std::array<T, N> &arr, const std::string_view &sep = ",") {
   std::string comma;
   std::string ret;
   for (size_t i = 0; i < N; i++) {
     ret += comma;
     ret += std::to_string(arr[i]);
     comma = sep;
   }
   return ret;
 }

 /** Result are appended strings with each prefixed by a byte of length
  */
 std::string to_mdns_string(const std::vector<std::string> &list);

 /** see to_string(first, last, sep)
  */
 std::string to_string(const std::vector<std::string> &list,
                       const std::string &sep = ", ");

 /** creates a string: [elt1, elt2, ...] if sep == ', '
  * elt1<sep>elt2<sep> otherwise
  */
 std::string to_string(const std::vector<std::string>::const_iterator &first,
                       const std::vector<std::string>::const_iterator &last,
                       const std::string &sep = ", ");

 template <typename K, typename V>
 std::string to_string(const std::map<K, V> &m) {
   std::string comma;
   std::string ret;
   ret = "{";
   for (const auto &it : m) {
     ret += comma;
     ret += to_string(it.first);
     ret += ":";
     ret += to_string(it.second);
     comma = ", ";
   }
   ret += "}";
   return ret;
 }

 template <typename K>
 std::string to_string(const std::optional<K> &m) {
   if (m.has_value()) {
     return to_string(m.value());
   }
   return "<None>";
 }

 std::vector<std::string> split(const std::string_view &s, const char sep);

 bool ends_with(const std::string_view &s, const std::string_view &end);

 std::optional<uint32_t> hex_string_to_int(const std::string &s);

 std::string trim(const std::string_view &s);

 std::optional<int32_t> parse_int(const std::string &value);

 std::string replace(const std::string_view &in, char from, char to);

 std::string remove(const std::string_view &in, char delete_char);

 std::string to_upper(const std::string &s);

 } // namespace StringUtils
#pragma once

#include <cassert>
#include <chrono>
#include <functional>

 namespace time_utils {
 /** atomic clock time since 1970-1-1
  */
 namespace tai {
 static constexpr auto TAI_SECS_PER_DAY = 86400ULL;
 static constexpr auto DAYS_PER_YEAR = 356ULL;
 static constexpr auto YEARS_DIFFERENCE_TO_PTP_EPOCH = 1972 - 1958;

 static constexpr auto NANOS_PER_MILLI = 1000ULL;
 static constexpr auto MILLIS_PER_SEC = 1000ULL;

 static constexpr auto MICROS_PER_SEC = 1000ULL * MILLIS_PER_SEC;
 static constexpr auto NANOS_PER_SEC = 1000ULL * MICROS_PER_SEC;

 class nanoseconds {
 public:
   explicit nanoseconds(uint64_t nanos) : m_nanos(nanos) {}
   uint64_t count() const { return m_nanos; }

   /** @returns TAI timestamp string: seconds:nanoseconds
    */
   std::string to_TAI_timestamp() const;

 private:
   uint64_t m_nanos;
 };

 class milliseconds {
 public:
   explicit milliseconds(uint64_t millis) : m_millis(millis) {}
   milliseconds(const nanoseconds &ns) : m_millis(ns.count() / NANOS_PER_MILLI) {}

   uint64_t count() const { return m_millis; }

 private:
   uint64_t m_millis;
 };

 class seconds {
 public:
   explicit seconds(uint64_t secs) : m_secs(secs) {}
   seconds(const milliseconds &ms) : m_secs(ms.count() / MILLIS_PER_SEC) {}
   seconds(const nanoseconds &ns) : m_secs(ns.count() / NANOS_PER_SEC) {}

   uint64_t count() const { return m_secs; }

 private:
   uint64_t m_secs;
 };

 static inline nanoseconds get_current_time() {
   // epoch in chrono::tai is 1958-01-01 00:00:00.
   // for PTP/SMPTE we need an epoch of  1970-01-01
   // THis means adding 12 years to the chono
   // For TAI a day is exactly  86,400 seconds.

   const auto tai_now = std::chrono::tai_clock::now().time_since_epoch();
   const auto secs_too_much = std::chrono::seconds(TAI_SECS_PER_DAY * DAYS_PER_YEAR * YEARS_DIFFERENCE_TO_PTP_EPOCH);
   const auto tai_adjusted = tai_now - secs_too_much;
   return nanoseconds(tai_adjusted.count());
 }
 } // namespace tai

 static inline std::chrono::nanoseconds get_current_time() {
#if 1
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC_RAW, &now);
   return std::chrono::nanoseconds(((uint64_t)now.tv_sec * UINT64_C(1000000000)) + (uint64_t)now.tv_nsec);
#else
   std::chrono::time_point<std::chrono::system_clock> now =
       std::chrono::system_clock::now();
   const auto duration = now.time_since_epoch();
   const auto nanos =
       std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
   return nanos;
#endif
 }

 static inline std::chrono::microseconds get_current_time_micros() {
   const auto now = get_current_time();
   return std::chrono::duration_cast<std::chrono::microseconds>(now);
 }

 static inline std::chrono::milliseconds get_current_time_millis() {
   const auto now = get_current_time();
   return std::chrono::duration_cast<std::chrono::milliseconds>(now);
 }

 class Timeout {
 public:
   Timeout(const std::chrono::microseconds &t) {
     m_deadline = get_current_time() + t;
   }

   bool elapsed() const {
     return get_current_time() >= m_deadline;
   }

   std::chrono::nanoseconds time_left() const {
     return m_deadline - get_current_time();
   }

 private:
   std::chrono::nanoseconds m_deadline;
 };

 } // namespace time_utils

 /**
  * @file IOUring.hpp
  * @brief Defines the IOUring class for asynchronous I/O operations using
  * io_uring.
  */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* See feature_test_macros(7) */
#endif

#include <netdb.h>

#include <liburing.h>

#include <expected>
#include <stack>

#include <slogger/Error.hpp>

#include "iuring/IOUringInterface.hpp"
#include "iuring/NetworkAdapter.hpp"

#include "WorkPool.hpp"

 static constexpr size_t DEFAULT_QUEUE_SIZE = 64;

 namespace iuring {
 class IOUring : public IOUringInterface,
                 public std::enable_shared_from_this<IOUring> {
 private:
   IOUring(
       logging::ILogger &logger, NetworkAdapter &adapter, size_t queue_size);

   IOUring(const IOUring &) = delete;
   IOUring &operator=(const IOUring &) = delete;
   IOUring(IOUring &&) = delete;
   IOUring &operator=(IOUring &&) = delete;

 public:
   static std::shared_ptr<IOUring> create(logging::ILogger &logger,
                                          NetworkAdapter &adapter, size_t queue_size = DEFAULT_QUEUE_SIZE);

   ~IOUring();

   error::Error init() override;

   error::Error poll_completion_queues() override;

   std::shared_ptr<IWorkItem> ackuire_send_workitem(
       const std::shared_ptr<ISocket> &socket) override;

   void submit_connect(const std::shared_ptr<ISocket> &socket,
                       const IPAddress &target, connect_callback_func_t handler) override;

   void submit_accept(const std::shared_ptr<ISocket> &socket,
                      accept_callback_func_t handler) override;

   void submit_recv(const std::shared_ptr<ISocket> &socket,
                    recv_callback_func_t handler) override;

   void submit_close(const std::shared_ptr<ISocket> &socket,
                     close_callback_func_t handler) override;

   void resolve_hostname(const std::string &hostname,
                         const resolve_hostname_callback_func_t &handler) override;

   NetworkAdapter &get_adapter() {
     return m_adapter;
   }

 private:
   static constexpr auto QD = 64;
   static constexpr auto BUF_SHIFT = 12; /* 4k */
   static constexpr auto CQES = (QD * 16);
   static constexpr auto BUFFERS = CQES;

   bool m_initialized = false;
   logging::ILogger &m_logger;
   size_t m_queue_size = 0;
   io_uring_buf_reg m_reg;

   io_uring m_ring{};
   io_uring_buf_ring *buf_ring = nullptr;
   size_t buf_ring_size = 0;
   size_t buf_shift = BUF_SHIFT;
   unsigned char *buffer_base = nullptr;
   std::stack<int> m_free_send_ids;

   NetworkAdapter &m_adapter;
   WorkPool m_pool;

   class RequestInfo {
   public:
     RequestInfo() = delete;
     RequestInfo(const RequestInfo &) = delete;
     RequestInfo &operator=(const RequestInfo &) = delete;
     RequestInfo &operator=(RequestInfo &&) = delete;

     RequestInfo(RequestInfo &&arg) {
       hostname = std::move(arg.hostname);
       handlers = std::move(arg.handlers);

       request = arg.request;
       all_requests[0] = request;

       arg.request = nullptr;
       arg.all_requests[0] = nullptr;
     }

     RequestInfo(const std::string &_hostname)
         : hostname(_hostname) {
     }

   public:
     std::string hostname;

     // for getaddrinfo_a
     gaicb *request = new gaicb{};
     gaicb *all_requests[1] = {request};

     // requests for this hostname:
     std::vector<IOUring::resolve_hostname_callback_func_t> handlers;
   };

   std::vector<RequestInfo> m_hostname_DNS_requests;

   logging::ILogger &get_logger() {
     return m_logger;
   }

   WorkPool &get_pool() {
     return m_pool;
   }

   error::Error setup_buffer_pool();
   void probe_features();
   void init_ring();

   void submit_all_requests();

   size_t buffer_size() const {
     assert(buf_shift > 0);
     return 1U << buf_shift;
   }

   uint8_t *get_buffer(int idx) {
     assert(buf_shift > 0);
     return buffer_base + (idx << buf_shift);
   }

   static void sig_notifier_hostname_resolve(sigval_t sv);
   void trigger_hostname_resolve_callbacks(void *ptr);

   void recycle_buffer(int idx);

   void submit(IWorkItem &item) override;

   void send_packet(const std::shared_ptr<WorkItem> &work_item);

   void call_callback_and_free_work_item_id(io_uring_cqe *cqe);

   io_uring_sqe *get_sqe();

   void call_send_callback(
       std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe);

   void call_close_callback(
       std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe);

   void call_connect_callback(
       std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe);

   ReceivePostAction call_recv_handler_stream(const uint8_t *buffer,
                                              std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe);

   ReceivePostAction call_recv_handler_datagram(const uint8_t *buffer,
                                                std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe);

   ReceivePostAction call_recv_callback(
       std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe);

   void call_accept_callback(
       std::shared_ptr<WorkItem> work_item, io_uring_cqe *cqe);
 };
 } // namespace iuring

 namespace iuring {
 class ProbeUringFeatures {
 public:
   UringFeature convert_uring_op_to_feature(int op) {
     switch (op) {
     case IORING_OP_NOP:
       return UringFeature::IORING_OP_NOP;
     case IORING_OP_READV:
       return UringFeature::IORING_OP_READV;
     case IORING_OP_WRITEV:
       return UringFeature::IORING_OP_WRITEV;
     case IORING_OP_FSYNC:
       return UringFeature::IORING_OP_FSYNC;
     case IORING_OP_READ_FIXED:
       return UringFeature::IORING_OP_READ_FIXED;
     case IORING_OP_WRITE_FIXED:
       return UringFeature::IORING_OP_WRITE_FIXED;
     case IORING_OP_POLL_ADD:
       return UringFeature::IORING_OP_POLL_ADD;
     case IORING_OP_POLL_REMOVE:
       return UringFeature::IORING_OP_POLL_REMOVE;
     case IORING_OP_SYNC_FILE_RANGE:
       return UringFeature::IORING_OP_SYNC_FILE_RANGE;
     case IORING_OP_SENDMSG:
       return UringFeature::IORING_OP_SENDMSG;
     case IORING_OP_RECVMSG:
       return UringFeature::IORING_OP_RECVMSG;
     case IORING_OP_TIMEOUT:
       return UringFeature::IORING_OP_TIMEOUT;
     case IORING_OP_TIMEOUT_REMOVE:
       return UringFeature::IORING_OP_TIMEOUT_REMOVE;
     case IORING_OP_ACCEPT:
       return UringFeature::IORING_OP_ACCEPT;
#if SUPPORT_LISTEN_IN_LIBURING
     case IORING_OP_LISTEN:
       return UringFeature::IORING_OP_LISTEN;
#endif
     case IORING_OP_ASYNC_CANCEL:
       return UringFeature::IORING_OP_ASYNC_CANCEL;
     case IORING_OP_LINK_TIMEOUT:
       return UringFeature::IORING_OP_LINK_TIMEOUT;
     case IORING_OP_CONNECT:
       return UringFeature::IORING_OP_CONNECT;
     case IORING_OP_FALLOCATE:
       return UringFeature::IORING_OP_FALLOCATE;
     case IORING_OP_OPENAT:
       return UringFeature::IORING_OP_OPENAT;
     case IORING_OP_CLOSE:
       return UringFeature::IORING_OP_CLOSE;
     case IORING_OP_FILES_UPDATE:
       return UringFeature::IORING_OP_FILES_UPDATE;
     case IORING_OP_STATX:
       return UringFeature::IORING_OP_STATX;
     case IORING_OP_READ:
       return UringFeature::IORING_OP_READ;
     case IORING_OP_WRITE:
       return UringFeature::IORING_OP_WRITE;
     case IORING_OP_FADVISE:
       return UringFeature::IORING_OP_FADVISE;
     case IORING_OP_MADVISE:
       return UringFeature::IORING_OP_MADVISE;
     case IORING_OP_SEND:
       return UringFeature::IORING_OP_SEND;
     case IORING_OP_RECV:
       return UringFeature::IORING_OP_RECV;
     case IORING_OP_OPENAT2:
       return UringFeature::IORING_OP_OPENAT2;
     case IORING_OP_EPOLL_CTL:
       return UringFeature::IORING_OP_EPOLL_CTL;
     case IORING_OP_SPLICE:
       return UringFeature::IORING_OP_SPLICE;
     case IORING_OP_PROVIDE_BUFFERS:
       return UringFeature::IORING_OP_PROVIDE_BUFFERS;
     case IORING_OP_REMOVE_BUFFERS:
       return UringFeature::IORING_OP_REMOVE_BUFFERS;
     case IORING_OP_TEE:
       return UringFeature::IORING_OP_TEE;
     case IORING_OP_SHUTDOWN:
       return UringFeature::IORING_OP_SHUTDOWN;
     case IORING_OP_RENAMEAT:
       return UringFeature::IORING_OP_RENAMEAT;
     case IORING_OP_UNLINKAT:
       return UringFeature::IORING_OP_UNLINKAT;
     case IORING_OP_MKDIRAT:
       return UringFeature::IORING_OP_MKDIRAT;
     case IORING_OP_SYMLINKAT:
       return UringFeature::IORING_OP_SYMLINKAT;
     case IORING_OP_LINKAT:
       return UringFeature::IORING_OP_LINKAT;
     }
     return UringFeature::UNKNOWN;
   }

   ProbeUringFeatures(
       io_uring *ring, logging::ILogger &logger)
       : m_logger(logger) {
     m_probe = io_uring_get_probe_ring(ring);
     if (!m_probe) {
       LOG_ERROR(m_logger, "failed to probe uring features\n");
       abort();
     }

     for (size_t i = 0; i < m_probe->ops_len; i++) {
       const auto op = m_probe->ops[i].op;
       const auto flags = m_probe->ops[i].flags;

       if (!(flags & IO_URING_OP_SUPPORTED)) {
         continue;
       }

       // fprintf(stderr, "supported: op: {}\n", op);
       const auto ec = convert_uring_op_to_feature(op);
       m_features[ec] = true;
     }
   }

   bool supports(UringFeature f) const {
     return m_features.contains(f);
   }

   ~ProbeUringFeatures() {
     io_uring_free_probe(m_probe);
   }

 private:
   std::map<UringFeature, bool> m_features;
   logging::ILogger &m_logger;
   io_uring_probe *m_probe;
 };

 } // namespace iuring
#pragma once

#include <net/if.h>
#include <netinet/in.h>
#include <optional>
#include <poll.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <memory>

#include <slogger/ILogger.hpp>
#include <slogger/StringUtils.hpp>

#include <iuring/IPAddress.hpp>
#include <iuring/ISocket.hpp>

 namespace iuring {
 class AcceptResult;

 class SocketImpl : public ISocket {
 private:
   SocketImpl(logging::ILogger &logger, const AcceptResult &new_conn);

   SocketImpl(
       SocketType type, SocketPortID port, logging::ILogger &logger, SocketKind kind);

 public:
   static std::shared_ptr<SocketImpl> create(logging::ILogger &logger, const AcceptResult &new_conn);
   static std::shared_ptr<SocketImpl> create(SocketType type, SocketPortID port, logging::ILogger &logger, SocketKind kind);

   void dump_info();

   int mcast_bind() override;

   void join_multicast_group(const std::string &ip_address,
                             const std::string &source_iface) override;
   void leave_multicast_group();

 private:
   ip_mreq m_mreq{};

   void local_bind(SocketPortID port_id);
 };

 } // namespace iuring

#include <arpa/inet.h>
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <variant>
#include <vector>

#include <liburing.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <iuring/ISocket.hpp>
#include <iuring/IWorkItem.hpp>
#include <iuring/NetworkProtocols.hpp>
#include <iuring/ReceivedMessage.hpp>
#include <iuring/SendPacket.hpp>
#include <iuring/UringDefs.hpp>

 namespace iuring {

 using work_item_id_t = uint64_t;

 class IOUringInterface;

 class WorkItem : public IWorkItem {
 public:
   WorkItem(logging::ILogger &logger,
            const std::shared_ptr<IOUringInterface> &network, work_item_id_t id,
            const char *descr, const std::shared_ptr<ISocket> &s)
       : m_logger(logger), m_io_ring(network), m_id(id), m_socket(s), m_descr(descr) {
   }

   bool is_free() const {
     return m_state == State::FREE;
   }

   void mark_in_use() {
     assert(m_state == State::FREE);
     m_state = State::IN_USE;
   }

   void mark_is_free() {
     assert(m_state == State::IN_USE);
     m_state = State::FREE;
   }

   static const char *type_to_string(Type t);

   const char *get_type_str() const {
     return type_to_string(get_type());
   }

   bool is_stream() const {
     return m_socket->is_stream();
   }

   /** submit a connect request */
   void submit(const IPAddress &target, const connect_callback_func_t &cb);
   /** submit a send request */
   void submit_stream_data(const send_callback_func_t &cb) override;
   /** submit a send request */
   void submit_packet(const DatagramSendParameters &params,
                      const send_callback_func_t &cb) override;
   /** submit a recv request */
   void submit(const recv_callback_func_t &cb);
   /** submit a accept request */
   void submit(const accept_callback_func_t &cb);
   /** submit a close request */
   void submit(const close_callback_func_t &cb);

   void clean_send_packet() {
     m_send_packet.reset();
   }

   std::shared_ptr<ISocket> get_socket() const override {
     return m_socket;
   }

   work_item_id_t get_id() const {
     return m_id;
   }

   void call_send_callback(int status) {
     assert(std::holds_alternative<send_callback_func_t>(m_callback));
     auto call = std::get<send_callback_func_t>(m_callback);
     SendResult result{status};
     call(result);
     m_send_packet.reset();
   }

   void call_close_callback(int status) {
     assert(std::holds_alternative<close_callback_func_t>(m_callback));
     auto call = std::get<close_callback_func_t>(m_callback);

     CloseResult result{status};
     call(result);
   }

   [[nodiscard]] ReceivePostAction call_recv_callback(
       const ReceivedMessage &payload) const {
     assert(std::holds_alternative<recv_callback_func_t>(m_callback));
     auto call = std::get<recv_callback_func_t>(m_callback);
     return call(payload);
   }

   void call_accept_callback(const AcceptResult &new_conn) const {
     assert(std::holds_alternative<accept_callback_func_t>(m_callback));
     auto call = std::get<accept_callback_func_t>(m_callback);
     call(new_conn);
   }

   void call_connect_callback(const ConnectResult &new_conn) const {
     assert(std::holds_alternative<connect_callback_func_t>(m_callback));
     auto call = std::get<connect_callback_func_t>(m_callback);
     call(new_conn);
   }

   SendPacket &get_send_packet() override {
     m_send_packet.reset();
     return m_send_packet;
   }

   const SendPacket &get_raw_send_packet() const {
     return m_send_packet;
   }

   bool is_recv_request() const {
     return m_work_type == Type::RECV;
   }

   const std::string &get_descr() const {
     return m_descr;
   }

   bool next_request_should_wait_for_this_request() const {
     return m_link_to_next_request;
   }

 private:
   enum class State {
     IN_USE,
     FREE
   };

   logging::ILogger &m_logger;
   DatagramSendParameters m_params;

   State m_state = State::IN_USE;

   SendPacket m_send_packet;
   std::shared_ptr<IOUringInterface> m_io_ring;
   work_item_id_t m_id;

   std::variant<connect_callback_func_t, accept_callback_func_t,
                recv_callback_func_t, send_callback_func_t, close_callback_func_t>
       m_callback;

   // used/set when creating submit entry:
   std::array<char, 1024> m_control;
   std::shared_ptr<ISocket> m_socket;
   msghdr m_msg;
   std::shared_ptr<std::array<iovec, 1>> m_msg_iov =
       std::make_shared<std::array<iovec, 1>>();
   IPAddress m_sa;
   sockaddr_storage m_buffer_for_uring;
   socklen_t m_accept_sock_len = 0;
   socklen_t m_connect_sock_len = 0;
   std::string m_descr;

   // if the next request should wait for this one to finish
   bool m_link_to_next_request = false;

   const IPAddress &get_socket_address() const {
     return m_sa;
   }

   logging::ILogger &get_logger() {
     return m_logger;
   }

   ReceivePostAction do_stream_socket_receive();
   ReceivePostAction do_packet_socket_receive();
   void init_send_msg();

   friend class IOUring;
 };

 SocketType get_type(const AcceptResult &res);
 SocketPortID get_port(const AcceptResult &res);

 } // namespace iuring

 // #define USE_PLAIN_SOCKETS 1

#include <arpa/inet.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <liburing.h>
#include <sys/mman.h>

#include <sys/socket.h>
#include <sys/types.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <vector>

#include <slogger/ILogger.hpp>

#include "WorkItem.hpp"
#include <iuring/UringDefs.hpp>

 namespace iuring {
 class IOUringInterface;

 class WorkPool {
 public:
   explicit WorkPool(logging::ILogger &logger)
       : m_logger(logger) {
   }

   logging::ILogger &get_logger() {
     return m_logger;
   }

   std::shared_ptr<WorkItem> alloc_send_work_item(
       const std::shared_ptr<ISocket> &socket,
       const std::shared_ptr<IOUringInterface> &network,
       const char *descr);

   std::shared_ptr<WorkItem> alloc_recv_work_item(
       const std::shared_ptr<ISocket> &socket,
       const std::shared_ptr<IOUringInterface> &network,
       const recv_callback_func_t &callback, const char *descr);

   std::shared_ptr<WorkItem> alloc_accept_work_item(
       const std::shared_ptr<ISocket> &socket,
       const std::shared_ptr<IOUringInterface> &network,
       const accept_callback_func_t &callback, const char *descr);

   std::shared_ptr<WorkItem> alloc_connect_work_item(
       const IPAddress &target,
       const std::shared_ptr<ISocket> &socket,
       const std::shared_ptr<IOUringInterface> &network,
       const connect_callback_func_t &callback, const char *descr);

   std::shared_ptr<WorkItem> alloc_close_work_item(
       const std::shared_ptr<ISocket> &socket,
       const std::shared_ptr<iuring::IOUringInterface> &network,
       const close_callback_func_t &callback, const char *descr);

   std::shared_ptr<WorkItem> get_work_item(work_item_id_t id);
   void free_work_item(work_item_id_t id);

   // only for testing:
   const std::vector<std::shared_ptr<WorkItem>> &get_work_item_list() const {
     return m_work_items;
   }

 private:
   logging::ILogger &m_logger;
   std::recursive_mutex m_mutex;
   /** because we pass indices into this array into io_uring, we are not
    * allowed to shrink this using erase(), we therefore append only and have
    * m_free_ids to track free indices.
    */
   std::vector<std::shared_ptr<WorkItem>> m_work_items;
   std::stack<work_item_id_t> m_free_ids;

   std::shared_ptr<WorkItem> internal_alloc_work_item(
       const std::shared_ptr<ISocket> &socket,
       const std::shared_ptr<IOUringInterface> &network, const char *descr);
 };

 } // namespace iuring

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iuring/IOUringInterface.hpp>
#include <iuring/ISocketFactory.hpp>

 namespace iuring::mocks {

 class Socket : public ISocket {
 public:
   Socket(SocketType type, SocketPortID port, logging::ILogger &logger,
          SocketKind kind, int fd)
       : ISocket(type, port, logger, kind, fd) {
   }

   MOCK_METHOD(int, mcast_bind, (), (override));
   MOCK_METHOD(void, join_multicast_group,
               (const std::string &ip_address, const std::string &source_iface),
               (override));
 };

 class SocketFactory : public ISocketFactory {
 public:
   std::shared_ptr<ISocket> create_impl(SocketType type, SocketPortID port,
                                        logging::ILogger &logger, SocketKind kind) override {
     // fd is not used for mocked sockets; use -1 as placeholder
     return std::make_shared<Socket>(type, port, logger, kind, -1);
   }

   std::shared_ptr<ISocket> create_impl(
       logging::ILogger &logger, const AcceptResult &res) override {
     // create a mock Socket reflecting the accepted connection
     const auto port = res.m_address.get_port();
     const auto type = SocketType::IPV4_TCP;
     return std::make_shared<Socket>(type, port, logger,
                                     SocketKind::SERVER_STREAM_SOCKET, res.m_new_fd);
   }
 };

 class WorkItem : public IWorkItem {
 public:
   MOCK_METHOD(SendPacket &, get_send_packet, (), (override));
   MOCK_METHOD(void, submit_packet, (const DatagramSendParameters &params, const send_callback_func_t &cb), (override));
   MOCK_METHOD(void, submit_stream_data, (const send_callback_func_t &cb), (override));
   MOCK_METHOD(std::shared_ptr<ISocket>, get_socket, (), (const, override));
 };

 class IOUring : public IOUringInterface {
 public:
   MOCK_METHOD(error::Error, init, (), (override));
   MOCK_METHOD(error::Error, poll_completion_queues, (), (override));
   MOCK_METHOD(void, submit_connect,
               (const std::shared_ptr<ISocket> &socket, const IPAddress &target,
                connect_callback_func_t handler),
               (override));
   MOCK_METHOD(void, submit_accept,
               (const std::shared_ptr<ISocket> &socket,
                accept_callback_func_t handler),
               (override));
   MOCK_METHOD(void, submit_recv,
               (const std::shared_ptr<ISocket> &socket,
                recv_callback_func_t handler),
               (override));
   MOCK_METHOD(std::shared_ptr<IWorkItem>, ackuire_send_workitem,
               (const std::shared_ptr<ISocket> &socket), (override));
   MOCK_METHOD(void, submit, (IWorkItem & item), (override));
   MOCK_METHOD(void, submit_close,
               (const std::shared_ptr<ISocket> &socket,
                close_callback_func_t handler),
               (override));

   MOCK_METHOD(void, resolve_hostname, (const std::string &hostname, const resolve_hostname_callback_func_t &handler), (override));
 };
 } // namespace iuring::mocks

#include <cassert>
#include <functional>
#include <string>
#include <vector>

#include <slogger/ILogger.hpp>
#include <slogger/TimeUtils.hpp>

#include "task_defs.hpp"

 namespace realtime {
 class RealtimeKernel;

 class BaseTask {
 public:
   BaseTask(TaskType tt, const std::string &name,
            const std::chrono::microseconds &t, task_func_t callback,
            logging::ILogger &logger, RealtimeKernel *kernel)
       : m_task_type(tt), m_interval(t), m_task_func(callback), m_timeout(std::chrono::microseconds(0)), m_name(name), m_logger(logger), m_kernel(kernel) {
     assert(kernel != nullptr);
   }

   TaskType get_task_type() const {
     return m_task_type;
   }

   logging::ILogger &get_logger() {
     return m_logger;
   }

   std::string get_service_status_as_json() const;

   const std::string &get_name() const {
     return m_name;
   }

   void run();

   /** called to wait for deadline to elapse because there's no more idle tasks
    * that we can squeeze into the time until this task needs to run.
    * We only need to do this for hard-realtime tasks as soft ones can run when
    * we have time for them afterwards.
    */
   void wait_for_deadline() const {
     assert(get_task_type() == TaskType::HARD_REALTIME);
     while (!m_timeout.elapsed()) {
       ;
     }
   }

   void run_elapsed() {
     if (get_task_type() == TaskType::HARD_REALTIME) {
       assert(m_timeout.elapsed());
     }

     m_timeout = time_utils::Timeout(m_interval);

     run();
   }

   bool have_time_left_before_deadline() const {
     return !m_timeout.elapsed();
   }

   std::chrono::nanoseconds time_left_until_deadline() const {
     assert(this != nullptr);
     return m_timeout.time_left();
   }

   /** if you need the most accuracy */
   std::chrono::nanoseconds max_time_taken_ns() const {
     return m_max_time_taken;
   }

   /** if you need the most accuracy */
   std::chrono::nanoseconds warmup_max_time_taken_ns() const {
     return m_warmup_max_time_taken;
   }

   std::chrono::microseconds max_time_taken_us() const {
     return std::chrono::duration_cast<std::chrono::microseconds>(m_max_time_taken);
   }

   std::chrono::microseconds warmup_max_time_taken_us() const {
     return std::chrono::duration_cast<std::chrono::microseconds>(m_warmup_max_time_taken);
   }

   std::chrono::nanoseconds average_time_taken_ns() const {
     return std::chrono::duration_cast<std::chrono::nanoseconds>(
         average_time_taken_us());
   }

   std::chrono::microseconds average_time_taken_us() const {
     return std::chrono::microseconds(
         (int64_t)((double)m_total_time_taken_us.count() / m_num_calls));
   }

   void set_period(const std::chrono::microseconds &t) {
     m_interval = t;
   }

   void enable() {
     m_enabled = true;
   }

   void disable() {
     m_enabled = false;
   }

   bool is_enabled() const {
     return m_enabled;
   }

 private:
   TaskType m_task_type;
   std::chrono::microseconds m_interval = std::chrono::microseconds(0);
   std::chrono::nanoseconds m_max_time_taken = std::chrono::nanoseconds(0);
   // to avoid overflow, lets use micros here:
   std::chrono::microseconds m_total_time_taken_us = std::chrono::microseconds(0);
   std::chrono::nanoseconds m_warmup_max_time_taken =
       std::chrono::nanoseconds(0);

   uint64_t m_num_calls = 0;
   uint64_t m_num_task_ok_calls = 0;
   task_func_t m_task_func;
   time_utils::Timeout m_timeout;
   bool m_enabled = false;
   std::string m_name;
   logging::ILogger &m_logger;
   RealtimeKernel *m_kernel = nullptr;
 };

 } // namespace realtime

#include <string>

#include <slogger/Error.hpp>

 namespace service {
 class IService {
 public:
   virtual std::string get_service_status_as_json() const { return ""; }

   [[nodiscard]]
   virtual error::Error init() = 0;

   [[nodiscard]]
   virtual error::Error finish() = 0;
 };
 } // namespace service
#pragma once

#include <chrono>
#include <string>

#include <slogger/ILogger.hpp>

#include "BaseTask.hpp"

 namespace realtime {

 /** Instances of these are created by the RealtimeKernel::add_idle() method.
  * They represent tasks that run when no periodic tasks are scheduled to run.
  */
 class IdleTask : public BaseTask {
 public:
   IdleTask(const std::string &name, const std::chrono::microseconds &t,
            task_func_t callback, logging::ILogger &logger, RealtimeKernel *kernel)
       : BaseTask(TaskType::SOFT_REALTIME, name, t, callback, logger, kernel) {
   }
 };
 } // namespace realtime

#include <memory>
#include <vector>

#include <slogger/ILogger.hpp>

#include <urtsched/RealtimeKernel.hpp>
#include <urtsched/ServiceBus.hpp>

 namespace realtime {

 class MultiCoreRealtimeKernel {
 public:
   MultiCoreRealtimeKernel(logging::ILogger &logger, service::ServiceBus &bus) : m_bus(bus), m_logger(logger) {}

   std::shared_ptr<RealtimeKernel> add_core() {
     auto k = std::make_shared<RealtimeKernel>(m_logger, "core-" + std::to_string(m_kernels.size()));
     m_bus.add(k);
     m_kernels.push_back(k);
     return k;
   }

   void run(const std::chrono::milliseconds &max_runtime);

   logging::ILogger &get_logger() const { return m_logger; }

 private:
   service::ServiceBus &m_bus;
   logging::ILogger &m_logger;

   // one per core:
   std::vector<std::shared_ptr<RealtimeKernel>> m_kernels;

   void reserve_cores();
 };

 } // namespace realtime
#pragma once

#include <chrono>

#include <slogger/ILogger.hpp>

#include "BaseTask.hpp"

 namespace realtime {

 /** Instances of these are created by the RealtimeKernel::add_periodic() method.
  * They represent tasks that run periodically at a defined interval.
  */
 class PeriodicTask : public BaseTask {
 public:
   PeriodicTask(TaskType tt, const std::string &name,
                const std::chrono::microseconds &t, task_func_t callback,
                logging::ILogger &logger, RealtimeKernel *kernel)
       : BaseTask(tt, name, t, callback, logger, kernel) {
   }

   bool overlaps_with(const PeriodicTask &other) const;
 };

 } // namespace realtime

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <slogger/ILogger.hpp>
#include <slogger/TimeUtils.hpp>

#include <urtsched/IService.hpp>
#include <urtsched/fixed_size_vector.hpp>

#include "BaseTask.hpp"
#include "IdleTask.hpp"
#include "PeriodicTask.hpp"

 namespace realtime {
 /** Schedules stuff on a single core.
  * As its for a single core only, it does not need
  * locks/synchronization code and is therefore really fast.
  * We do not support between code task migration as a consequence.
  */
 class RealtimeKernel : public service::IService {
 public:
   RealtimeKernel(logging::ILogger &logger, const std::string &name)
       : m_logger(logger), m_name(name) {
   }

   [[nodiscard]] error::Error init() override {
     return error::Error::OK;
   }

   [[nodiscard]] error::Error finish() override {
     return error::Error::OK;
   }

   /** returns the name of the kernel */
   const std::string &get_name() const { return m_name; }

   /** Add a periodic task to the scheduler.
    * The returned task is disabled by default.
    * Therefore, call periodic->enable() to enable it.
    */
   [[nodiscard]] std::shared_ptr<PeriodicTask> add_periodic(
       TaskType tt,
       const std::string &name, const std::chrono::microseconds &interval,
       const task_func_t &callback);

   /** Add an idle task to the scheduler.
    * It is enabled by default.
    */
   [[nodiscard]] std::shared_ptr<IdleTask> add_idle_task(
       const std::string &name, const task_func_t &callback);

   /** return true on successful removal */
   bool remove(const std::shared_ptr<PeriodicTask> &task_ptr);

   bool should_exit() const {
     return false;
   }

   /** @param max_runtime if 0 return forever
    */
   void run(const std::chrono::milliseconds &max_runtime);

   void step();

   std::string get_service_status_as_json() const;

   logging::ILogger &get_logger() {
     return m_logger;
   }

 private:
   static constexpr bool m_debug = false;
   logging::ILogger &m_logger;
   const std::string m_name;

   static constexpr auto MAX_PERIODIC_TASKS = 64;
   static constexpr auto MAX_IDLE_TASKS = 16;

   realtime::fixed_size_vector<std::shared_ptr<PeriodicTask>, MAX_PERIODIC_TASKS> m_periodic_list;
   realtime::fixed_size_vector<std::shared_ptr<IdleTask>, MAX_IDLE_TASKS> m_idle_list;

   std::vector<std::shared_ptr<PeriodicTask>> get_next_periodics();

   // return the task thats earliest:
   std::shared_ptr<PeriodicTask> get_earliest_next_periodic();

   /**return a list of tasks whose runtime can overlap 'next'.
    * the returned list contains 'next' as well.
    */
   std::vector<std::shared_ptr<PeriodicTask>> get_periodics_that_can_overlap(const std::shared_ptr<PeriodicTask> &next);

   /** return a sorted list of real-time tasks */
   std::vector<std::shared_ptr<PeriodicTask>> get_sorted_realtime_tasks(const std::vector<std::shared_ptr<PeriodicTask>> &next_up);

   void run_idle_tasks();
 };

 } // namespace realtime

#include <map>
#include <memory>
#include <optional>
#include <queue>

#include <slogger/ILogger.hpp>

#include <urtsched/RealtimeKernel.hpp>

 namespace service {
 class Service : public IService {
 public:
   Service(const std::shared_ptr<realtime::RealtimeKernel> &rt_kernel,
           logging::ILogger &logger)
       : m_rt_kernel(rt_kernel), m_logger(logger) {
   }

   logging::ILogger &get_logger() {
     return m_logger;
   }

   std::shared_ptr<realtime::RealtimeKernel> &get_rt_kernel() {
     return m_rt_kernel;
   }

   /** if there's nothing to do, call the work pushed with run_oneshot_idle_task()
    */
   void run_oneshot_idle_task(const std::string &name, const realtime::task_func_t &f);

 private:
   std::shared_ptr<realtime::RealtimeKernel> m_rt_kernel;
   logging::ILogger &m_logger;

   std::map<std::string, std::shared_ptr<realtime::IdleTask>> m_tasks;
 };
 } // namespace service
#pragma once

#include <memory>
#include <vector>

#include <urtsched/IService.hpp>

 namespace service {

 /** all services are registered in this bus
  * and can be queried for their status info
  */
 class ServiceBus {
 public:
   std::string get_service_status_as_json();

   void add(const std::shared_ptr<service::IService> s) {
     m_services.push_back(s);
   }

 private:
   std::vector<std::shared_ptr<service::IService>> m_services;
 };
 } // namespace service

#include <array>
#include <memory>
#include <stdexcept>

#include <iterator>

 namespace realtime {

 /** a vector with a maximum size so we're sure that its not resized even when
  * pushing back */
 template <typename T, size_t N>
 class fixed_size_vector {
 public:
   // Iterator type definitions
   using value_type = T;
   using size_type = size_t;
   using difference_type = std::ptrdiff_t;
   using reference = T &;
   using const_reference = const T &;
   using pointer = T *;
   using const_pointer = const T *;
   using iterator = typename std::array<T, N>::iterator;
   using const_iterator = typename std::array<T, N>::const_iterator;
   using reverse_iterator = std::reverse_iterator<iterator>;
   using const_reverse_iterator = std::reverse_iterator<const_iterator>;

   size_t size() const {
     return m_size;
   }

   bool empty() const {
     return m_size == 0;
   }

   size_t capacity() const {
     return N;
   }

   void push_back(const T &value) {
     if (m_size >= N) {
       throw std::runtime_error("fixed_size_vector overflow");
     }
     m_data[m_size++] = value;
   }

   void clear() {
     m_size = 0;
   }

   // Element access
   reference operator[](size_type pos) {
     return m_data[pos];
   }

   const_reference operator[](size_type pos) const {
     return m_data[pos];
   }

   reference at(size_type pos) {
     if (pos >= m_size) {
       throw std::out_of_range("fixed_size_vector::at");
     }
     return m_data[pos];
   }

   const_reference at(size_type pos) const {
     if (pos >= m_size) {
       throw std::out_of_range("fixed_size_vector::at");
     }
     return m_data[pos];
   }

   reference front() {
     return m_data[0];
   }

   const_reference front() const {
     return m_data[0];
   }

   reference back() {
     return m_data[m_size - 1];
   }

   const_reference back() const {
     return m_data[m_size - 1];
   }

   pointer data() {
     return m_data.data();
   }

   const_pointer data() const {
     return m_data.data();
   }

   // Iterator support
   iterator begin() {
     return m_data.begin();
   }

   const_iterator begin() const {
     return m_data.begin();
   }

   const_iterator cbegin() const {
     return m_data.cbegin();
   }

   iterator end() {
     return m_data.begin() + m_size;
   }

   const_iterator end() const {
     return m_data.begin() + m_size;
   }

   const_iterator cend() const {
     return m_data.cbegin() + m_size;
   }

   reverse_iterator rbegin() {
     return reverse_iterator(end());
   }

   const_reverse_iterator rbegin() const {
     return const_reverse_iterator(end());
   }

   const_reverse_iterator crbegin() const {
     return const_reverse_iterator(cend());
   }

   reverse_iterator rend() {
     return reverse_iterator(begin());
   }

   const_reverse_iterator rend() const {
     return const_reverse_iterator(begin());
   }

   const_reverse_iterator crend() const {
     return const_reverse_iterator(cbegin());
   }

 private:
   std::array<T, N> m_data;
   size_t m_size = 0;
 };

 } // namespace realtime
#pragma once

 namespace realtime {
 enum class TaskStatus {
   TASK_OK,
   TASK_YIELD
 };

 class BaseTask;

 using task_func_t = std::function<TaskStatus(BaseTask &)>;

 enum class TaskType {
   HARD_REALTIME,
   SOFT_REALTIME
 };

 } // namespace realtime

#include <format>
#include <string>

 /**
  * @file Error.hpp
  * @brief Defines error codes and conversion functions for network operations.
  */

 namespace error {

 enum class Error {
   OK,
   RANGE,
   MMAP_FAILED,
   FAILED_TO_CREATE_SOCKET,
   FAILED_TO_OPEN_PCM,
   ALSA_FAILURE,
   NO_ALSA_CAPTURE,
   NO_ALSA_PLAYBACK,
   HOSTNAME_RESOLVE_FAILED,
   BAD_PPROTOCOL,
   INVALID_SDP,
   UNKNOWN
 };

 Error errno_to_error(int err);
 } // namespace error

 template <>
 struct std::formatter<error::Error> {
   constexpr auto parse(std::format_parse_context &ctx) {
     return ctx.begin();
   }

   auto format(const error::Error &c, std::format_context &ctx) const {
     switch (c) {
     case error::Error::OK:
       return std::format_to(ctx.out(), "OK");
     case error::Error::RANGE:
       return std::format_to(ctx.out(), "RANGE");
     case error::Error::MMAP_FAILED:
       return std::format_to(ctx.out(), "MMAP_FAILED");
     case error::Error::FAILED_TO_CREATE_SOCKET:
       return std::format_to(ctx.out(), "FAILED_TO_CREATE_SOCKET");
     case error::Error::FAILED_TO_OPEN_PCM:
       return std::format_to(ctx.out(), "FAILED_TO_OPEN_PCM");
     case error::Error::ALSA_FAILURE:
       return std::format_to(ctx.out(), "ALSA_FAILURE");
     case error::Error::NO_ALSA_CAPTURE:
       return std::format_to(ctx.out(), "NO_ALSA_CAPTURE");
     case error::Error::NO_ALSA_PLAYBACK:
       return std::format_to(ctx.out(), "NO_ALSA_PLAYBACK");
     case error::Error::HOSTNAME_RESOLVE_FAILED:
       return std::format_to(ctx.out(), "HOSTNAME_RESOLVE_FAILED");
     case error::Error::UNKNOWN:
       return std::format_to(ctx.out(), "UNKNOWN");
     case error::Error::BAD_PPROTOCOL:
       return std::format_to(ctx.out(), "BAD_PPROTOCOL");
     case error::Error::INVALID_SDP:
       return std::format_to(ctx.out(), "INVALID_SDP");
     }
     return std::format_to(ctx.out(), "UNKNOWN_ERROR_CODE");
   }
 };
#pragma once

 /**
  * @file ILogger.hpp
  * @brief Defines the ILogger interface for logging messages with different
  * severity levels.
  *
  * This interface provides methods for logging debug, info, and error messages.
  */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include <format>

 namespace logging {
 class ILogger {
 public:
   ILogger(bool debug, bool info)
       : m_debug(debug), m_info(info) {
   }

   virtual ~ILogger() {}

   virtual void debug_msg(
       uint32_t line, const char *file, const std::string msg) = 0;
   virtual void info_msg(
       uint32_t line, const char *file, const std::string msg) = 0;
   virtual void error_msg(
       uint32_t line, const char *file, const std::string msg) = 0;

   bool enable_debug() const {
     return m_debug;
   }
   bool enable_info() const {
     return m_info;
   }

 protected:
   bool m_debug = true;
   bool m_info = true;
 };

 const char *strip_prefix(const char *path);

#define LOG_DEBUG(logger, ...)                                  \
   if (logger.enable_debug()) {                                  \
     logger.debug_msg(__LINE__, logging::strip_prefix(__FILE__), \
                      std::format(__VA_ARGS__));                 \
   }

#define LOG_INFO(logger, ...)                                  \
   if (logger.enable_info()) {                                  \
     logger.info_msg(__LINE__, logging::strip_prefix(__FILE__), \
                     std::format(__VA_ARGS__));                 \
   }

#define LOG_INFO_ONCE(logger, ...)                               \
   do {                                                           \
     static bool init_once;                                       \
     if (logger.enable_info() && !init_once) {                    \
       logger.info_msg(__LINE__, logging::strip_prefix(__FILE__), \
                       std::format(__VA_ARGS__));                 \
       init_once = true;                                          \
     }                                                            \
   } while (0)

#define LOG_ERROR(logger, ...) \
   logger.error_msg(            \
       __LINE__, logging::strip_prefix(__FILE__), std::format(__VA_ARGS__))

#define LOG_ERROR_ONCE(logger, ...)                               \
   do {                                                            \
     static bool init_once;                                        \
     if (!init_once) {                                             \
       logger.error_msg(__LINE__, logging::strip_prefix(__FILE__), \
                        std::format(__VA_ARGS__));                 \
       init_once = true;                                           \
     }                                                             \
   } while (0)

#define LOG_ERROR_SLOW(logger, ...)                               \
   do {                                                            \
     static uint32_t counter;                                      \
     if (counter++ > 1000) {                                       \
       logger.error_msg(__LINE__, logging::strip_prefix(__FILE__), \
                        std::format(__VA_ARGS__));                 \
       counter = 0;                                                \
     }                                                             \
   } while (0)

 } // namespace logging

 /**
  * @file Logger.hpp
  * @brief Defines the Logger class for logging messages with different severity
  * levels.
  *
  * This class provides methods for logging debug, info, and error messages.
  * It supports logging to the console or to a file stream.
  */

#include <stdint.h>
#include <stdio.h>
#include <string>

#include "ILogger.hpp"

 namespace logging {

 enum class LogOutput {
   CONSOLE,
   FILE_STREAM
 };

 class Logger : public ILogger {
 public:
   void debug_msg(uint32_t line, const char *file, const std::string msg) override {
     log(line, file, Level::DEBUG, msg);
   }

   void info_msg(
       uint32_t line, const char *file, const std::string msg) override {
     log(line, file, Level::INFO, msg);
   }

   void error_msg(
       uint32_t line, const char *file, const std::string msg) override {
     log(line, file, Level::ERROR, msg);
   }

   Logger(bool debug, bool info, LogOutput output);
   ~Logger();

 private:
   LogOutput m_output;
   FILE *m_f = nullptr;

   enum class Level {
     DEBUG,
     INFO,
     ERROR
   };

   void log(
       uint32_t line, const char *file, Level level, const std::string &msg);
 };

 } // namespace logging

#include <string>

#include <slogger/ILogger.hpp>

 namespace shell {
 enum class RunOpt {
   LOG_ERROR_AS_WARNING,
   ABORT_ON_ERROR
 };

 void run_cmd(const std::string &cmd, logging::ILogger &logger, RunOpt opt);
 } // namespace shell

#include <array>
#include <map>
#include <optional>
#include <stdint.h>
#include <string>
#include <vector>

 /**
  * note: these functions are functional: they return the result and do not
  * change their arguments
  */

 namespace StringUtils {
 /** We get name lists like: [foo, local], then check if the last equals "local"
  */
 bool last_item_equals(const std::vector<std::string> &name_list, const std::string &last_item_name);

 template <typename T>
 bool contains(const std::vector<T> &l, const T elt) {
   for (auto it : l) {
     if (it == elt) {
       return true;
     }
   }
   return false;
 }

 [[maybe_unused]]
 static inline std::string to_string(bool v) {
   return v ? "true" : "false";
 }

 [[maybe_unused]]
 static inline std::string to_string(int v) {
   return std::to_string(v);
 }

 [[maybe_unused]]
 static inline std::string to_string(const std::string &v) {
   return v;
 }

 /** @returns comma seperated string
  */
 template <typename T, size_t N>
 std::string array_to_string(const std::array<T, N> &arr, const std::string_view &sep = ",") {
   std::string comma;
   std::string ret;
   for (size_t i = 0; i < N; i++) {
     ret += comma;
     ret += std::to_string(arr[i]);
     comma = sep;
   }
   return ret;
 }

 /** Result are appended strings with each prefixed by a byte of length
  */
 std::string to_mdns_string(const std::vector<std::string> &list);

 /** see to_string(first, last, sep)
  */
 std::string to_string(const std::vector<std::string> &list,
                       const std::string &sep = ", ");

 /** creates a string: [elt1, elt2, ...] if sep == ', '
  * elt1<sep>elt2<sep> otherwise
  */
 std::string to_string(const std::vector<std::string>::const_iterator &first,
                       const std::vector<std::string>::const_iterator &last,
                       const std::string &sep = ", ");

 template <typename K, typename V>
 std::string to_string(const std::map<K, V> &m) {
   std::string comma;
   std::string ret;
   ret = "{";
   for (const auto &it : m) {
     ret += comma;
     ret += to_string(it.first);
     ret += ":";
     ret += to_string(it.second);
     comma = ", ";
   }
   ret += "}";
   return ret;
 }

 template <typename K>
 std::string to_string(const std::optional<K> &m) {
   if (m.has_value()) {
     return to_string(m.value());
   }
   return "<None>";
 }

 std::vector<std::string> split(const std::string_view &s, const char sep);

 bool ends_with(const std::string_view &s, const std::string_view &end);

 std::optional<uint32_t> hex_string_to_int(const std::string &s);

 std::string trim(const std::string_view &s);

 std::optional<int32_t> parse_int(const std::string &value);

 std::string replace(const std::string_view &in, char from, char to);

 std::string remove(const std::string_view &in, char delete_char);

 std::string to_upper(const std::string &s);

 } // namespace StringUtils
#pragma once

#include <cassert>
#include <chrono>
#include <functional>

 namespace time_utils {
 /** atomic clock time since 1970-1-1
  */
 namespace tai {
 static constexpr auto TAI_SECS_PER_DAY = 86400ULL;
 static constexpr auto DAYS_PER_YEAR = 356ULL;
 static constexpr auto YEARS_DIFFERENCE_TO_PTP_EPOCH = 1972 - 1958;

 static constexpr auto NANOS_PER_MILLI = 1000ULL;
 static constexpr auto MILLIS_PER_SEC = 1000ULL;

 static constexpr auto MICROS_PER_SEC = 1000ULL * MILLIS_PER_SEC;
 static constexpr auto NANOS_PER_SEC = 1000ULL * MICROS_PER_SEC;

 class nanoseconds {
 public:
   explicit nanoseconds(uint64_t nanos) : m_nanos(nanos) {}
   uint64_t count() const { return m_nanos; }

   /** @returns TAI timestamp string: seconds:nanoseconds
    */
   std::string to_TAI_timestamp() const;

 private:
   uint64_t m_nanos;
 };

 class milliseconds {
 public:
   explicit milliseconds(uint64_t millis) : m_millis(millis) {}
   milliseconds(const nanoseconds &ns) : m_millis(ns.count() / NANOS_PER_MILLI) {}

   uint64_t count() const { return m_millis; }

 private:
   uint64_t m_millis;
 };

 class seconds {
 public:
   explicit seconds(uint64_t secs) : m_secs(secs) {}
   seconds(const milliseconds &ms) : m_secs(ms.count() / MILLIS_PER_SEC) {}
   seconds(const nanoseconds &ns) : m_secs(ns.count() / NANOS_PER_SEC) {}

   uint64_t count() const { return m_secs; }

 private:
   uint64_t m_secs;
 };

 static inline nanoseconds get_current_time() {
   // epoch in chrono::tai is 1958-01-01 00:00:00.
   // for PTP/SMPTE we need an epoch of  1970-01-01
   // THis means adding 12 years to the chono
   // For TAI a day is exactly  86,400 seconds.

   const auto tai_now = std::chrono::tai_clock::now().time_since_epoch();
   const auto secs_too_much = std::chrono::seconds(TAI_SECS_PER_DAY * DAYS_PER_YEAR * YEARS_DIFFERENCE_TO_PTP_EPOCH);
   const auto tai_adjusted = tai_now - secs_too_much;
   return nanoseconds(tai_adjusted.count());
 }
 } // namespace tai

 static inline std::chrono::nanoseconds get_current_time() {
#if 1
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC_RAW, &now);
   return std::chrono::nanoseconds(((uint64_t)now.tv_sec * UINT64_C(1000000000)) + (uint64_t)now.tv_nsec);
#else
   std::chrono::time_point<std::chrono::system_clock> now =
       std::chrono::system_clock::now();
   const auto duration = now.time_since_epoch();
   const auto nanos =
       std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
   return nanos;
#endif
 }

 static inline std::chrono::microseconds get_current_time_micros() {
   const auto now = get_current_time();
   return std::chrono::duration_cast<std::chrono::microseconds>(now);
 }

 static inline std::chrono::milliseconds get_current_time_millis() {
   const auto now = get_current_time();
   return std::chrono::duration_cast<std::chrono::milliseconds>(now);
 }

 class Timeout {
 public:
   Timeout(const std::chrono::microseconds &t) {
     m_deadline = get_current_time() + t;
   }

   bool elapsed() const {
     return get_current_time() >= m_deadline;
   }

   std::chrono::nanoseconds time_left() const {
     return m_deadline - get_current_time();
   }

 private:
   std::chrono::nanoseconds m_deadline;
 };

 } // namespace time_utils