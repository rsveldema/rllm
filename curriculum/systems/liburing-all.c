/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_PROXY_H
#define LIBURING_PROXY_H
#include <sys/time.h>
/*
* Generic opcode agnostic encoding to sqe/cqe->user_data
*/
struct userdata {
union {
struct {
uint16_t op_tid;
/* 4 bits op, 12 bits tid */
uint16_t bid;
uint16_t fd;
};
uint64_t val;
};
};
#define OP_SHIFT (12)
#define TID_MASK ((1U << 12) - 1)
/*
* Packs the information that we will need at completion time into the
* sqe->user_data field, which is passed back in the completion in
* cqe->user_data. Some apps would need more space than this, and in fact
* I'd love to pack the requested IO size in here, and it's not uncommon to
* see apps use this field as just a cookie to either index a data structure
* at completion time, or even just put the pointer to the associated
* structure into this field.
*/
static inline void __encode_userdata(struct io_uring_sqe *sqe, int tid, int op,
int bid, int fd){
struct userdata ud = {
.op_tid = (op << OP_SHIFT) | tid,
.bid = bid,
.fd = fd
};
io_uring_sqe_set_data64(sqe, ud.val);
}
static inline uint64_t __raw_encode(int tid, int op, int bid, int fd){
struct userdata ud = {
.op_tid = (op << OP_SHIFT) | tid,
.bid = bid,
.fd = fd
};
return ud.val;
}
static inline int cqe_to_op(struct io_uring_cqe *cqe){
struct userdata ud = { .val = cqe->user_data };
return ud.op_tid >> OP_SHIFT;
}
static inline int cqe_to_bid(struct io_uring_cqe *cqe){
struct userdata ud = { .val = cqe->user_data };
return ud.bid;
}
static inline int cqe_to_fd(struct io_uring_cqe *cqe){
struct userdata ud = { .val = cqe->user_data };
return ud.fd;
}
static unsigned long long mtime_since(const struct timeval *s,
const struct timeval *e){
long long sec, usec;
sec = e->tv_sec - s->tv_sec;
usec = (e->tv_usec - s->tv_usec);
if (sec > 0 && usec < 0) {
sec--;
usec += 1000000;
}
sec *= 1000;
usec /= 1000;
return sec + usec;
}
static unsigned long long mtime_since_now(struct timeval *tv){
struct timeval end;
gettimeofday(&end, NULL);
return mtime_since(tv, &end);
}
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_EX_HELPERS_H
#define LIBURING_EX_HELPERS_H
#include <stddef.h>
#define T_ALIGN_UP(v, align) (((v) + (align) - 1) & ~((align) - 1))
int setup_listening_socket(int port, int ipv6);
/*
* Some Android versions lack aligned_alloc in stdlib.h.
* To avoid making large changes in tests, define a helper
* function that wraps posix_memalign as our own aligned_alloc.
*/
void *t_aligned_alloc(size_t alignment, size_t size);
void t_error(int status, int errnum, const char *format, ...);
#ifndef CONFIG_HAVE_MEMFD_CREATE
#include <linux/memfd.h>
#endif
int memfd_create(const char *name, unsigned int flags);
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_AARCH64_SYSCALL_H
#define LIBURING_ARCH_AARCH64_SYSCALL_H
#if defined(__aarch64__)
#define __do_syscallN(...) ({ \
__asm__ volatile ( \
"svc 0" \
: "=r"(x0) \
: __VA_ARGS__ \
: "memory", "cc"); \
(long) x0; \
})
#define __do_syscall0(__n) ({ \
register long x8 __asm__("x8") = __n; \
register long x0 __asm__("x0"); \
\
__do_syscallN("r" (x8)); \
})
#define __do_syscall1(__n, __a) ({ \
register long x8 __asm__("x8") = __n; \
register __typeof__(__a) x0 __asm__("x0") = __a; \
\
__do_syscallN("r" (x8), "0" (x0)); \
})
#define __do_syscall2(__n, __a, __b) ({ \
register long x8 __asm__("x8") = __n; \
register __typeof__(__a) x0 __asm__("x0") = __a; \
register __typeof__(__b) x1 __asm__("x1") = __b; \
\
__do_syscallN("r" (x8), "0" (x0), "r" (x1)); \
})
#define __do_syscall3(__n, __a, __b, __c) ({ \
register long x8 __asm__("x8") = __n; \
register __typeof__(__a) x0 __asm__("x0") = __a; \
register __typeof__(__b) x1 __asm__("x1") = __b; \
register __typeof__(__c) x2 __asm__("x2") = __c; \
\
__do_syscallN("r" (x8), "0" (x0), "r" (x1), "r" (x2)); \
})
#define __do_syscall4(__n, __a, __b, __c, __d) ({ \
register long x8 __asm__("x8") = __n; \
register __typeof__(__a) x0 __asm__("x0") = __a; \
register __typeof__(__b) x1 __asm__("x1") = __b; \
register __typeof__(__c) x2 __asm__("x2") = __c; \
register __typeof__(__d) x3 __asm__("x3") = __d; \
\
__do_syscallN("r" (x8), "0" (x0), "r" (x1), "r" (x2), "r" (x3));\
})
#define __do_syscall5(__n, __a, __b, __c, __d, __e) ({ \
register long x8 __asm__("x8") = __n; \
register __typeof__(__a) x0 __asm__("x0") = __a; \
register __typeof__(__b) x1 __asm__("x1") = __b; \
register __typeof__(__c) x2 __asm__("x2") = __c; \
register __typeof__(__d) x3 __asm__("x3") = __d; \
register __typeof__(__e) x4 __asm__("x4") = __e; \
\
__do_syscallN("r" (x8), "0" (x0), "r" (x1), "r" (x2), "r" (x3), \
"r"(x4)); \
})
#define __do_syscall6(__n, __a, __b, __c, __d, __e, __f) ({ \
register long x8 __asm__("x8") = __n; \
register __typeof__(__a) x0 __asm__("x0") = __a; \
register __typeof__(__b) x1 __asm__("x1") = __b; \
register __typeof__(__c) x2 __asm__("x2") = __c; \
register __typeof__(__d) x3 __asm__("x3") = __d; \
register __typeof__(__e) x4 __asm__("x4") = __e; \
register __typeof__(__f) x5 __asm__("x5") = __f; \
\
__do_syscallN("r" (x8), "0" (x0), "r" (x1), "r" (x2), "r" (x3), \
"r" (x4), "r"(x5)); \
})
#include "../syscall-defs.h"
#else
/* #if defined(__aarch64__) */
#include "../generic/syscall.h"
#endif
/* #if defined(__aarch64__) */
#endif
/* #ifndef LIBURING_ARCH_AARCH64_SYSCALL_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_AARCH64_LIB_H
#define LIBURING_ARCH_AARCH64_LIB_H
#include <elf.h>
#include "../../syscall.h"
#ifndef CONFIG_NOLIBC
#include <unistd.h>
static inline long __get_page_size(void){
long ret = sysconf(_SC_PAGESIZE);
if (ret < 0)
ret = 4096;
return ret;
}
#else
static inline long __get_page_size(void){
Elf64_Off buf[2];
long ret = 4096;
int fd;
fd = __sys_open("/proc/self/auxv", O_RDONLY, 0);
if (fd < 0)
return ret;
while (1) {
ssize_t x;
x = __sys_read(fd, buf, sizeof(buf));
if (x < (long) sizeof(buf))
break;
if (buf[0] == AT_PAGESZ) {
ret = buf[1];
break;
}
}
__sys_close(fd);
return ret;
}
#endif
static inline long get_page_size(void){
static long cache_val;
if (cache_val)
return cache_val;
cache_val = __get_page_size();
return cache_val;
}
#endif
/* #ifndef LIBURING_ARCH_AARCH64_LIB_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_GENERIC_LIB_H
#define LIBURING_ARCH_GENERIC_LIB_H
static inline long get_page_size(void){
long page_size;
page_size = sysconf(_SC_PAGESIZE);
if (page_size < 0)
page_size = 4096;
return page_size;
}
#endif
/* #ifndef LIBURING_ARCH_GENERIC_LIB_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_GENERIC_SYSCALL_H
#define LIBURING_ARCH_GENERIC_SYSCALL_H
#include <fcntl.h>
static inline int __sys_io_uring_register(unsigned int fd, unsigned int opcode,
const void *arg, unsigned int nr_args){
int ret;
ret = syscall(__NR_io_uring_register, fd, opcode, arg, nr_args);
return (ret < 0) ? -errno : ret;
}
static inline int __sys_io_uring_setup(unsigned int entries,
struct io_uring_params *p){
int ret;
ret = syscall(__NR_io_uring_setup, entries, p);
return (ret < 0) ? -errno : ret;
}
static inline int __sys_io_uring_enter2(unsigned int fd, unsigned int to_submit,
unsigned int min_complete,
unsigned int flags, void *arg,
size_t sz){
int ret;
ret = syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags,
arg, sz);
return (ret < 0) ? -errno : ret;
}
static inline int __sys_io_uring_enter(unsigned int fd, unsigned int to_submit,
unsigned int min_complete,
unsigned int flags, sigset_t *sig){
return __sys_io_uring_enter2(fd, to_submit, min_complete, flags, sig,
_NSIG / 8);
}
static inline int __sys_open(const char *pathname, int flags, mode_t mode){
int ret;
ret = open(pathname, flags, mode);
return (ret < 0) ? -errno : ret;
}
static inline ssize_t __sys_read(int fd, void *buffer, size_t size){
ssize_t ret;
ret = read(fd, buffer, size);
return (ret < 0) ? -errno : ret;
}
static inline void *__sys_mmap(void *addr, size_t length, int prot, int flags,
int fd, off_t offset){
void *ret;
ret = mmap(addr, length, prot, flags, fd, offset);
return (ret == MAP_FAILED) ? ERR_PTR(-errno) : ret;
}
static inline int __sys_munmap(void *addr, size_t length){
int ret;
ret = munmap(addr, length);
return (ret < 0) ? -errno : ret;
}
static inline int __sys_madvise(void *addr, size_t length, int advice){
int ret;
ret = madvise(addr, length, advice);
return (ret < 0) ? -errno : ret;
}
static inline int __sys_getrlimit(int resource, struct rlimit *rlim){
int ret;
ret = getrlimit(resource, rlim);
return (ret < 0) ? -errno : ret;
}
static inline int __sys_setrlimit(int resource, const struct rlimit *rlim){
int ret;
ret = setrlimit(resource, rlim);
return (ret < 0) ? -errno : ret;
}
static inline int __sys_close(int fd){
int ret;
ret = close(fd);
return (ret < 0) ? -errno : ret;
}
#endif
/* #ifndef LIBURING_ARCH_GENERIC_SYSCALL_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_RISCV64_SYSCALL_H
#define LIBURING_ARCH_RISCV64_SYSCALL_H
#if defined(__riscv) && __riscv_xlen == 64
#define __do_syscallM(...) ({ \
__asm__ volatile ( \
"ecall" \
: "=r"(a0) \
: __VA_ARGS__ \
: "memory", "a1"); \
(long) a0; \
})
#define __do_syscallN(...) ({ \
__asm__ volatile ( \
"ecall" \
: "=r"(a0) \
: __VA_ARGS__ \
: "memory"); \
(long) a0; \
})
#define __do_syscall0(__n) ({ \
register long a7 __asm__("a7") = __n; \
register long a0 __asm__("a0"); \
\
__do_syscallM("r" (a7)); \
})
#define __do_syscall1(__n, __a) ({ \
register long a7 __asm__("a7") = __n; \
register __typeof__(__a) a0 __asm__("a0") = __a; \
\
__do_syscallM("r" (a7), "0" (a0)); \
})
#define __do_syscall2(__n, __a, __b) ({ \
register long a7 __asm__("a7") = __n; \
register __typeof__(__a) a0 __asm__("a0") = __a; \
register __typeof__(__b) a1 __asm__("a1") = __b; \
\
__do_syscallN("r" (a7), "0" (a0), "r" (a1)); \
})
#define __do_syscall3(__n, __a, __b, __c) ({ \
register long a7 __asm__("a7") = __n; \
register __typeof__(__a) a0 __asm__("a0") = __a; \
register __typeof__(__b) a1 __asm__("a1") = __b; \
register __typeof__(__c) a2 __asm__("a2") = __c; \
\
__do_syscallN("r" (a7), "0" (a0), "r" (a1), "r" (a2)); \
})
#define __do_syscall4(__n, __a, __b, __c, __d) ({ \
register long a7 __asm__("a7") = __n; \
register __typeof__(__a) a0 __asm__("a0") = __a; \
register __typeof__(__b) a1 __asm__("a1") = __b; \
register __typeof__(__c) a2 __asm__("a2") = __c; \
register __typeof__(__d) a3 __asm__("a3") = __d; \
\
__do_syscallN("r" (a7), "0" (a0), "r" (a1), "r" (a2), "r" (a3));\
})
#define __do_syscall5(__n, __a, __b, __c, __d, __e) ({ \
register long a7 __asm__("a7") = __n; \
register __typeof__(__a) a0 __asm__("a0") = __a; \
register __typeof__(__b) a1 __asm__("a1") = __b; \
register __typeof__(__c) a2 __asm__("a2") = __c; \
register __typeof__(__d) a3 __asm__("a3") = __d; \
register __typeof__(__e) a4 __asm__("a4") = __e; \
\
__do_syscallN("r" (a7), "0" (a0), "r" (a1), "r" (a2), "r" (a3), \
"r"(a4)); \
})
#define __do_syscall6(__n, __a, __b, __c, __d, __e, __f) ({ \
register long a7 __asm__("a7") = __n; \
register __typeof__(__a) a0 __asm__("a0") = __a; \
register __typeof__(__b) a1 __asm__("a1") = __b; \
register __typeof__(__c) a2 __asm__("a2") = __c; \
register __typeof__(__d) a3 __asm__("a3") = __d; \
register __typeof__(__e) a4 __asm__("a4") = __e; \
register __typeof__(__f) a5 __asm__("a5") = __f; \
\
__do_syscallN("r" (a7), "0" (a0), "r" (a1), "r" (a2), "r" (a3), \
"r" (a4), "r"(a5)); \
})
#include "../syscall-defs.h"
#else
/* #if defined(__riscv) && __riscv_xlen == 64 */
#include "../generic/syscall.h"
#endif
/* #if defined(__riscv) && __riscv_xlen == 64 */
#endif
/* #ifndef LIBURING_ARCH_RISCV64_SYSCALL_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_RISCV64_LIB_H
#define LIBURING_ARCH_RISCV64_LIB_H
#include <elf.h>
#include <sys/auxv.h>
#include "../../syscall.h"
#ifndef CONFIG_NOLIBC
#include <unistd.h>
static inline long __get_page_size(void){
long ret = sysconf(_SC_PAGESIZE);
if (ret < 0)
ret = 4096;
return ret;
}
#else
static inline long __get_page_size(void){
Elf64_Off buf[2];
long ret = 4096;
int fd;
fd = __sys_open("/proc/self/auxv", O_RDONLY, 0);
if (fd < 0)
return ret;
while (1) {
ssize_t x;
x = __sys_read(fd, buf, sizeof(buf));
if (x < (long) sizeof(buf))
break;
if (buf[0] == AT_PAGESZ) {
ret = buf[1];
break;
}
}
__sys_close(fd);
return ret;
}
#endif
static inline long get_page_size(void){
static long cache_val;
if (cache_val)
return cache_val;
cache_val = __get_page_size();
return cache_val;
}
#endif
/* #ifndef LIBURING_ARCH_RISCV64_LIB_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_X86_LIB_H
#define LIBURING_ARCH_X86_LIB_H
static inline long get_page_size(void){
return 4096;
}
#endif
/* #ifndef LIBURING_ARCH_X86_LIB_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_X86_SYSCALL_H
#define LIBURING_ARCH_X86_SYSCALL_H
#if defined(__x86_64__)
/**
* Note for syscall registers usage (x86-64):
* - %rax is the syscall number.
* - %rax is also the return value.
* - %rdi is the 1st argument.
* - %rsi is the 2nd argument.
* - %rdx is the 3rd argument.
* - %r10 is the 4th argument (**yes it's %r10, not %rcx!**).
* - %r8 is the 5th argument.
* - %r9 is the 6th argument.
*
* `syscall` instruction will clobber %r11 and %rcx.
*
* After the syscall returns to userspace:
* - %r11 will contain %rflags.
* - %rcx will contain the return address.
*
* IOW, after the syscall returns to userspace:
* %r11 == %rflags and %rcx == %rip.
*/
#define __do_syscall0(NUM) ({ \
intptr_t rax; \
\
__asm__ volatile( \
"syscall" \
: "=a"(rax)
/* %rax */ \
: "a"(NUM)
/* %rax */ \
: "rcx", "r11", "memory" \
); \
rax; \
})
#define __do_syscall1(NUM, ARG1) ({ \
intptr_t rax; \
\
__asm__ volatile( \
"syscall" \
: "=a"(rax)
/* %rax */ \
: "a"((NUM)),
/* %rax */ \
"D"((ARG1))
/* %rdi */ \
: "rcx", "r11", "memory" \
); \
rax; \
})
#define __do_syscall2(NUM, ARG1, ARG2) ({ \
intptr_t rax; \
\
__asm__ volatile( \
"syscall" \
: "=a"(rax)
/* %rax */ \
: "a"((NUM)),
/* %rax */ \
"D"((ARG1)),
/* %rdi */ \
"S"((ARG2))
/* %rsi */ \
: "rcx", "r11", "memory" \
); \
rax; \
})
#define __do_syscall3(NUM, ARG1, ARG2, ARG3) ({ \
intptr_t rax; \
\
__asm__ volatile( \
"syscall" \
: "=a"(rax)
/* %rax */ \
: "a"((NUM)),
/* %rax */ \
"D"((ARG1)),
/* %rdi */ \
"S"((ARG2)),
/* %rsi */ \
"d"((ARG3))
/* %rdx */ \
: "rcx", "r11", "memory" \
); \
rax; \
})
#define __do_syscall4(NUM, ARG1, ARG2, ARG3, ARG4) ({ \
intptr_t rax; \
register __typeof__(ARG4) __r10 __asm__("r10") = (ARG4); \
\
__asm__ volatile( \
"syscall" \
: "=a"(rax)
/* %rax */ \
: "a"((NUM)),
/* %rax */ \
"D"((ARG1)),
/* %rdi */ \
"S"((ARG2)),
/* %rsi */ \
"d"((ARG3)),
/* %rdx */ \
"r"(__r10)
/* %r10 */ \
: "rcx", "r11", "memory" \
); \
rax; \
})
#define __do_syscall5(NUM, ARG1, ARG2, ARG3, ARG4, ARG5) ({ \
intptr_t rax; \
register __typeof__(ARG4) __r10 __asm__("r10") = (ARG4); \
register __typeof__(ARG5) __r8 __asm__("r8") = (ARG5); \
\
__asm__ volatile( \
"syscall" \
: "=a"(rax)
/* %rax */ \
: "a"((NUM)),
/* %rax */ \
"D"((ARG1)),
/* %rdi */ \
"S"((ARG2)),
/* %rsi */ \
"d"((ARG3)),
/* %rdx */ \
"r"(__r10),
/* %r10 */ \
"r"(__r8)
/* %r8 */ \
: "rcx", "r11", "memory" \
); \
rax; \
})
#define __do_syscall6(NUM, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6) ({ \
intptr_t rax; \
register __typeof__(ARG4) __r10 __asm__("r10") = (ARG4); \
register __typeof__(ARG5) __r8 __asm__("r8") = (ARG5); \
register __typeof__(ARG6) __r9 __asm__("r9") = (ARG6); \
\
__asm__ volatile( \
"syscall" \
: "=a"(rax)
/* %rax */ \
: "a"((NUM)),
/* %rax */ \
"D"((ARG1)),
/* %rdi */ \
"S"((ARG2)),
/* %rsi */ \
"d"((ARG3)),
/* %rdx */ \
"r"(__r10),
/* %r10 */ \
"r"(__r8),
/* %r8 */ \
"r"(__r9)
/* %r9 */ \
: "rcx", "r11", "memory" \
); \
rax; \
})
#include "../syscall-defs.h"
#else
/* #if defined(__x86_64__) */
#ifdef CONFIG_NOLIBC
/**
* Note for syscall registers usage (x86, 32-bit):
* - %eax is the syscall number.
* - %eax is also the return value.
* - %ebx is the 1st argument.
* - %ecx is the 2nd argument.
* - %edx is the 3rd argument.
* - %esi is the 4th argument.
* - %edi is the 5th argument.
* - %ebp is the 6th argument.
*/
#define __do_syscall0(NUM) ({ \
intptr_t eax; \
\
__asm__ volatile( \
"int $0x80" \
: "=a"(eax)
/* %eax */ \
: "a"(NUM)
/* %eax */ \
: "memory" \
); \
eax; \
})
#define __do_syscall1(NUM, ARG1) ({ \
intptr_t eax; \
\
__asm__ volatile( \
"int $0x80" \
: "=a"(eax)
/* %eax */ \
: "a"(NUM),
/* %eax */ \
"b"((ARG1))
/* %ebx */ \
: "memory" \
); \
eax; \
})
#define __do_syscall2(NUM, ARG1, ARG2) ({ \
intptr_t eax; \
\
__asm__ volatile( \
"int $0x80" \
: "=a" (eax)
/* %eax */ \
: "a"(NUM),
/* %eax */ \
"b"((ARG1)),
/* %ebx */ \
"c"((ARG2))
/* %ecx */ \
: "memory" \
); \
eax; \
})
#define __do_syscall3(NUM, ARG1, ARG2, ARG3) ({ \
intptr_t eax; \
\
__asm__ volatile( \
"int $0x80" \
: "=a" (eax)
/* %eax */ \
: "a"(NUM),
/* %eax */ \
"b"((ARG1)),
/* %ebx */ \
"c"((ARG2)),
/* %ecx */ \
"d"((ARG3))
/* %edx */ \
: "memory" \
); \
eax; \
})
#define __do_syscall4(NUM, ARG1, ARG2, ARG3, ARG4) ({ \
intptr_t eax; \
\
__asm__ volatile( \
"int $0x80" \
: "=a" (eax)
/* %eax */ \
: "a"(NUM),
/* %eax */ \
"b"((ARG1)),
/* %ebx */ \
"c"((ARG2)),
/* %ecx */ \
"d"((ARG3)),
/* %edx */ \
"S"((ARG4))
/* %esi */ \
: "memory" \
); \
eax; \
})
#define __do_syscall5(NUM, ARG1, ARG2, ARG3, ARG4, ARG5) ({ \
intptr_t eax; \
\
__asm__ volatile( \
"int $0x80" \
: "=a" (eax)
/* %eax */ \
: "a"(NUM),
/* %eax */ \
"b"((ARG1)),
/* %ebx */ \
"c"((ARG2)),
/* %ecx */ \
"d"((ARG3)),
/* %edx */ \
"S"((ARG4)),
/* %esi */ \
"D"((ARG5))
/* %edi */ \
: "memory" \
); \
eax; \
})
/*
* On i386, the 6th argument of syscall goes in %ebp. However, both Clang
* and GCC cannot use %ebp in the clobber list and in the "r" constraint
* without using -fomit-frame-pointer. To make it always available for
* any kind of compilation, the below workaround is implemented:
*
* 1) Push the 6-th argument.
* 2) Push %ebp.
* 3) Load the 6-th argument from 4(%esp) to %ebp.
* 4) Do the syscall (int $0x80).
* 5) Pop %ebp (restore the old value of %ebp).
* 6) Add %esp by 4 (undo the stack pointer).
*
* WARNING:
* Don't use register variables for __do_syscall6(), there is a known
* GCC bug that results in an endless loop.
*
* BugLink: https:
//gcc.gnu.org/bugzilla/show_bug.cgi?id=105032
*
*/
#define __do_syscall6(NUM, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6) ({ \
intptr_t eax = (intptr_t)(NUM); \
intptr_t arg6 = (intptr_t)(ARG6);
/* Always in memory */ \
__asm__ volatile ( \
"pushl %[_arg6]\n\t" \
"pushl %%ebp\n\t" \
"movl 4(%%esp),%%ebp\n\t" \
"int $0x80\n\t" \
"popl %%ebp\n\t" \
"addl $4,%%esp" \
: "+a"(eax)
/* %eax */ \
: "b"(ARG1),
/* %ebx */ \
"c"(ARG2),
/* %ecx */ \
"d"(ARG3),
/* %edx */ \
"S"(ARG4),
/* %esi */ \
"D"(ARG5),
/* %edi */ \
[_arg6]"m"(arg6)
/* memory */ \
: "memory", "cc" \
); \
eax; \
})
#include "../syscall-defs.h"
#else
/* #ifdef CONFIG_NOLIBC */
#include "../generic/syscall.h"
#endif
/* #ifdef CONFIG_NOLIBC */
#endif
/* #if defined(__x86_64__) */
#endif
/* #ifndef LIBURING_ARCH_X86_SYSCALL_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_ARCH_SYSCALL_DEFS_H
#define LIBURING_ARCH_SYSCALL_DEFS_H
#include <fcntl.h>
static inline int __sys_open(const char *pathname, int flags, mode_t mode){
/*
* Some architectures don't have __NR_open, but __NR_openat.
*/
#ifdef __NR_open
return (int) __do_syscall3(__NR_open, pathname, flags, mode);
#else
return (int) __do_syscall4(__NR_openat, AT_FDCWD, pathname, flags, mode);
#endif
}
static inline ssize_t __sys_read(int fd, void *buffer, size_t size){
return (ssize_t) __do_syscall3(__NR_read, fd, buffer, size);
}
static inline void *__sys_mmap(void *addr, size_t length, int prot, int flags,
int fd, off_t offset){
int nr;
#if defined(__NR_mmap2)
nr = __NR_mmap2;
offset >>= 12;
#else
nr = __NR_mmap;
#endif
return (void *) __do_syscall6(nr, addr, length, prot, flags, fd, offset);
}
static inline int __sys_munmap(void *addr, size_t length){
return (int) __do_syscall2(__NR_munmap, addr, length);
}
static inline int __sys_madvise(void *addr, size_t length, int advice){
return (int) __do_syscall3(__NR_madvise, addr, length, advice);
}
static inline int __sys_getrlimit(int resource, struct rlimit *rlim){
return (int) __do_syscall4(__NR_prlimit64, 0, resource, NULL, rlim);
}
static inline int __sys_setrlimit(int resource, const struct rlimit *rlim){
return (int) __do_syscall4(__NR_prlimit64, 0, resource, rlim, NULL);
}
static inline int __sys_close(int fd){
return (int) __do_syscall1(__NR_close, fd);
}
static inline int __sys_io_uring_register(unsigned int fd, unsigned int opcode,
const void *arg, unsigned int nr_args){
return (int) __do_syscall4(__NR_io_uring_register, fd, opcode, arg,
nr_args);
}
static inline int __sys_io_uring_setup(unsigned int entries,
struct io_uring_params *p){
return (int) __do_syscall2(__NR_io_uring_setup, entries, p);
}
static inline int __sys_io_uring_enter2(unsigned int fd, unsigned int to_submit,
unsigned int min_complete,
unsigned int flags, void *arg,
size_t sz){
return (int) __do_syscall6(__NR_io_uring_enter, fd, to_submit,
min_complete, flags, arg, sz);
}
static inline int __sys_io_uring_enter(unsigned int fd, unsigned int to_submit,
unsigned int min_complete,
unsigned int flags, sigset_t *sig){
return __sys_io_uring_enter2(fd, to_submit, min_complete, flags, sig,
_NSIG / 8);
}
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_BARRIER_H
#define LIBURING_BARRIER_H
/*
From the kernel documentation file refcount-vs-atomic.rst:
A RELEASE memory ordering guarantees that all prior loads and
stores (all po-earlier instructions) on the same CPU are completed
before the operation. It also guarantees that all po-earlier
stores on the same CPU and all propagated stores from other CPUs
must propagate to all other CPUs before the release operation
(A-cumulative property). This is implemented using
:c:func:`smp_store_release`.
An ACQUIRE memory ordering guarantees that all post loads and
stores (all po-later instructions) on the same CPU are
completed after the acquire operation. It also guarantees that all
po-later stores on the same CPU must propagate to all other CPUs
after the acquire operation executes. This is implemented using
:c:func:`smp_acquire__after_ctrl_dep`.
*/
#ifdef __cplusplus
#include <atomic>
#define LIBURING_NOEXCEPT noexcept
template <typename T>
_LOCAL_INLINE void IO_URING_WRITE_ONCE(T &var, T val)
LIBURING_NOEXCEPT{
std::atomic_store_explicit(reinterpret_cast<std::atomic<T> *>(&var),
val, std::memory_order_relaxed);
}
template <typename T>
_LOCAL_INLINE T IO_URING_READ_ONCE(const T &var)
LIBURING_NOEXCEPT{
return std::atomic_load_explicit(
reinterpret_cast<const std::atomic<T> *>(&var),
std::memory_order_relaxed);
}
template <typename T>
_LOCAL_INLINE void io_uring_smp_store_release(T *p, T v)
LIBURING_NOEXCEPT{
std::atomic_store_explicit(reinterpret_cast<std::atomic<T> *>(p), v,
std::memory_order_release);
}
template <typename T>
_LOCAL_INLINE T io_uring_smp_load_acquire(const T *p)
LIBURING_NOEXCEPT{
return std::atomic_load_explicit(
reinterpret_cast<const std::atomic<T> *>(p),
std::memory_order_acquire);
}
_LOCAL_INLINE void io_uring_smp_mb()
LIBURING_NOEXCEPT{
std::atomic_thread_fence(std::memory_order_seq_cst);
}
#else
#include <stdatomic.h>
#define IO_URING_WRITE_ONCE(var, val) \
atomic_store_explicit((_Atomic __typeof__(var) *)&(var), \
(val), memory_order_relaxed)
#define IO_URING_READ_ONCE(var) \
atomic_load_explicit((_Atomic __typeof__(var) *)&(var), \
memory_order_relaxed)
#define io_uring_smp_store_release(p, v) \
atomic_store_explicit((_Atomic __typeof__(*(p)) *)(p), (v), \
memory_order_release)
#define io_uring_smp_load_acquire(p) \
atomic_load_explicit((_Atomic __typeof__(*(p)) *)(p), \
memory_order_acquire)
#define io_uring_smp_mb() \
atomic_thread_fence(memory_order_seq_cst)
#endif
#endif
/* defined(LIBURING_BARRIER_H) */
/* SPDX-License-Identifier: (GPL-2.0 WITH Linux-syscall-note) OR MIT */
/*
* Header file for the io_uring interface.
*
* Copyright (C) 2019 Jens Axboe
* Copyright (C) 2019 Christoph Hellwig
*/
#ifndef LINUX_IO_URING_H
#define LINUX_IO_URING_H
#include <linux/fs.h>
#include <linux/types.h>
/*
* this file is shared with liburing and that has to autodetect
* if linux/time_types.h is available or not, it can
* define UAPI_LINUX_IO_URING_H_SKIP_LINUX_TIME_TYPES_H
* if linux/time_types.h is not available
*/
#ifndef UAPI_LINUX_IO_URING_H_SKIP_LINUX_TIME_TYPES_H
#include <linux/time_types.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif
/*
* IO submission data structure (Submission Queue Entry)
*/
struct io_uring_sqe {
__u8 opcode;
/* type of operation for this sqe */
__u8 flags;
/* IOSQE_ flags */
__u16 ioprio;
/* ioprio for the request */
__s32 fd;
/* file descriptor to do IO on */
union {
__u64 off;
/* offset into file */
__u64 addr2;
struct {
__u32 cmd_op;
__u32 __pad1;
};
};
union {
__u64 addr;
/* pointer to buffer or iovecs */
__u64 splice_off_in;
struct {
__u32 level;
__u32 optname;
};
};
__u32 len;
/* buffer size or number of iovecs */
union {
__kernel_rwf_t rw_flags;
__u32 fsync_flags;
__u16 poll_events;
/* compatibility */
__u32 poll32_events;
/* word-reversed for BE */
__u32 sync_range_flags;
__u32 msg_flags;
__u32 timeout_flags;
__u32 accept_flags;
__u32 cancel_flags;
__u32 open_flags;
__u32 statx_flags;
__u32 fadvise_advice;
__u32 splice_flags;
__u32 rename_flags;
__u32 unlink_flags;
__u32 hardlink_flags;
__u32 xattr_flags;
__u32 msg_ring_flags;
__u32 uring_cmd_flags;
__u32 waitid_flags;
__u32 futex_flags;
__u32 install_fd_flags;
__u32 nop_flags;
__u32 pipe_flags;
};
__u64 user_data;
/* data to be passed back at completion time */
/* pack this to avoid bogus arm OABI complaints */
union {
/* index into fixed buffers, if used */
__u16 buf_index;
/* for grouped buffer selection */
__u16 buf_group;
} __attribute__((packed));
/* personality to use, if used */
__u16 personality;
union {
__s32 splice_fd_in;
__u32 file_index;
__u32 zcrx_ifq_idx;
__u32 optlen;
struct {
__u16 addr_len;
__u16 __pad3[1];
};
};
union {
struct {
__u64 addr3;
__u64 __pad2[1];
};
struct {
__u64 attr_ptr;
/* pointer to attribute information */
__u64 attr_type_mask;
/* bit mask of attributes */
};
__u64 optval;
/*
* If the ring is initialized with IORING_SETUP_SQE128, then
* this field is used for 80 bytes of arbitrary command data
*/
__u8 cmd[0];
};
};
/* sqe->attr_type_mask flags */
#define IORING_RW_ATTR_FLAG_PI (1U << 0)
/* PI attribute information */
struct io_uring_attr_pi {
__u16 flags;
__u16 app_tag;
__u32 len;
__u64 addr;
__u64 seed;
__u64 rsvd;
};
/*
* If sqe->file_index is set to this for opcodes that instantiate a new
* direct descriptor (like openat/openat2/accept), then io_uring will allocate
* an available direct descriptor instead of having the application pass one
* in. The picked direct descriptor will be returned in cqe->res, or -ENFILE
* if the space is full.
*/
#define IORING_FILE_INDEX_ALLOC (~0U)
enum io_uring_sqe_flags_bit {
IOSQE_FIXED_FILE_BIT,
IOSQE_IO_DRAIN_BIT,
IOSQE_IO_LINK_BIT,
IOSQE_IO_HARDLINK_BIT,
IOSQE_ASYNC_BIT,
IOSQE_BUFFER_SELECT_BIT,
IOSQE_CQE_SKIP_SUCCESS_BIT,
};
/*
* sqe->flags
*/
/* use fixed fileset */
#define IOSQE_FIXED_FILE (1U << IOSQE_FIXED_FILE_BIT)
/* issue after inflight IO */
#define IOSQE_IO_DRAIN (1U << IOSQE_IO_DRAIN_BIT)
/* links next sqe */
#define IOSQE_IO_LINK (1U << IOSQE_IO_LINK_BIT)
/* like LINK, but stronger */
#define IOSQE_IO_HARDLINK (1U << IOSQE_IO_HARDLINK_BIT)
/* always go async */
#define IOSQE_ASYNC (1U << IOSQE_ASYNC_BIT)
/* select buffer from sqe->buf_group */
#define IOSQE_BUFFER_SELECT (1U << IOSQE_BUFFER_SELECT_BIT)
/* don't post CQE if request succeeded */
#define IOSQE_CQE_SKIP_SUCCESS (1U << IOSQE_CQE_SKIP_SUCCESS_BIT)
/*
* io_uring_setup() flags
*/
#define IORING_SETUP_IOPOLL (1U << 0)
/* io_context is polled */
#define IORING_SETUP_SQPOLL (1U << 1)
/* SQ poll thread */
#define IORING_SETUP_SQ_AFF (1U << 2)
/* sq_thread_cpu is valid */
#define IORING_SETUP_CQSIZE (1U << 3)
/* app defines CQ size */
#define IORING_SETUP_CLAMP (1U << 4)
/* clamp SQ/CQ ring sizes */
#define IORING_SETUP_ATTACH_WQ (1U << 5)
/* attach to existing wq */
#define IORING_SETUP_R_DISABLED (1U << 6)
/* start with ring disabled */
#define IORING_SETUP_SUBMIT_ALL (1U << 7)
/* continue submit on error */
/*
* Cooperative task running. When requests complete, they often require
* forcing the submitter to transition to the kernel to complete. If this
* flag is set, work will be done when the task transitions anyway, rather
* than force an inter-processor interrupt reschedule. This avoids interrupting
* a task running in userspace, and saves an IPI.
*/
#define IORING_SETUP_COOP_TASKRUN (1U << 8)
/*
* If COOP_TASKRUN is set, get notified if task work is available for
* running and a kernel transition would be needed to run it. This sets
* IORING_SQ_TASKRUN in the sq ring flags. Not valid without COOP_TASKRUN
* or DEFER_TASKRUN.
*/
#define IORING_SETUP_TASKRUN_FLAG (1U << 9)
#define IORING_SETUP_SQE128 (1U << 10)
/* SQEs are 128 byte */
#define IORING_SETUP_CQE32 (1U << 11)
/* CQEs are 32 byte */
/*
* Only one task is allowed to submit requests
*/
#define IORING_SETUP_SINGLE_ISSUER (1U << 12)
/*
* Defer running task work to get events.
* Rather than running bits of task work whenever the task transitions
* try to do it just before it is needed.
*/
#define IORING_SETUP_DEFER_TASKRUN (1U << 13)
/*
* Application provides the memory for the rings
*/
#define IORING_SETUP_NO_MMAP (1U << 14)
/*
* Register the ring fd in itself for use with
* IORING_REGISTER_USE_REGISTERED_RING; return a registered fd index rather
* than an fd.
*/
#define IORING_SETUP_REGISTERED_FD_ONLY (1U << 15)
/*
* Removes indirection through the SQ index array.
*/
#define IORING_SETUP_NO_SQARRAY (1U << 16)
/* Use hybrid poll in iopoll process */
#define IORING_SETUP_HYBRID_IOPOLL (1U << 17)
/*
* Allow both 16b and 32b CQEs. If a 32b CQE is posted, it will have
* IORING_CQE_F_32 set in cqe->flags.
*/
#define IORING_SETUP_CQE_MIXED (1U << 18)
/*
* Allow both 64b and 128b SQEs. If a 128b SQE is posted, it will use a 128b
* opcode.
*/
#define IORING_SETUP_SQE_MIXED (1U << 19)
/*
* When set, io_uring ignores SQ head and tail and fetches SQEs to submit
* starting from index 0 instead from the index stored in the head pointer.
* IOW, the user should place all SQE at the beginning of the SQ memory
* before issuing a submission syscall.
*
* It requires IORING_SETUP_NO_SQARRAY and is incompatible with
* IORING_SETUP_SQPOLL. The user must also never change the SQ head and tail
* values and keep it set to 0. Any other value is undefined behaviour.
*/
#define IORING_SETUP_SQ_REWIND (1U << 20)
enum io_uring_op {
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
IORING_OP_MSG_RING,
IORING_OP_FSETXATTR,
IORING_OP_SETXATTR,
IORING_OP_FGETXATTR,
IORING_OP_GETXATTR,
IORING_OP_SOCKET,
IORING_OP_URING_CMD,
IORING_OP_SEND_ZC,
IORING_OP_SENDMSG_ZC,
IORING_OP_READ_MULTISHOT,
IORING_OP_WAITID,
IORING_OP_FUTEX_WAIT,
IORING_OP_FUTEX_WAKE,
IORING_OP_FUTEX_WAITV,
IORING_OP_FIXED_FD_INSTALL,
IORING_OP_FTRUNCATE,
IORING_OP_BIND,
IORING_OP_LISTEN,
IORING_OP_RECV_ZC,
IORING_OP_EPOLL_WAIT,
IORING_OP_READV_FIXED,
IORING_OP_WRITEV_FIXED,
IORING_OP_PIPE,
IORING_OP_NOP128,
IORING_OP_URING_CMD128,
/* this goes last, obviously */
IORING_OP_LAST,
};
/*
* sqe->uring_cmd_flags top 8bits aren't available for userspace
* IORING_URING_CMD_FIXED use registered buffer; pass this flag
* along with setting sqe->buf_index.
*/
#define IORING_URING_CMD_FIXED (1U << 0)
#define IORING_URING_CMD_MASK IORING_URING_CMD_FIXED
/*
* sqe->fsync_flags
*/
#define IORING_FSYNC_DATASYNC (1U << 0)
/*
* sqe->timeout_flags
*
* IORING_TIMEOUT_IMMEDIATE_ARG: If set, sqe->addr stores the timeout
* value in nanoseconds instead of
* pointing to a timespec.
*/
#define IORING_TIMEOUT_ABS (1U << 0)
#define IORING_TIMEOUT_UPDATE (1U << 1)
#define IORING_TIMEOUT_BOOTTIME (1U << 2)
#define IORING_TIMEOUT_REALTIME (1U << 3)
#define IORING_LINK_TIMEOUT_UPDATE (1U << 4)
#define IORING_TIMEOUT_ETIME_SUCCESS (1U << 5)
#define IORING_TIMEOUT_MULTISHOT (1U << 6)
#define IORING_TIMEOUT_IMMEDIATE_ARG (1U << 7)
#define IORING_TIMEOUT_CLOCK_MASK (IORING_TIMEOUT_BOOTTIME | IORING_TIMEOUT_REALTIME)
#define IORING_TIMEOUT_UPDATE_MASK (IORING_TIMEOUT_UPDATE | IORING_LINK_TIMEOUT_UPDATE)
/*
* sqe->splice_flags
* extends splice(2) flags
*/
#define SPLICE_F_FD_IN_FIXED (1U << 31)
/* the last bit of __u32 */
/*
* POLL_ADD flags. Note that since sqe->poll_events is the flag space, the
* command flags for POLL_ADD are stored in sqe->len.
*
* IORING_POLL_ADD_MULTI Multishot poll. Sets IORING_CQE_F_MORE if
* the poll handler will continue to report
* CQEs on behalf of the same SQE.
*
* IORING_POLL_UPDATE Update existing poll request, matching
* sqe->addr as the old user_data field.
*
* IORING_POLL_LEVEL Level triggered poll.
*/
#define IORING_POLL_ADD_MULTI (1U << 0)
#define IORING_POLL_UPDATE_EVENTS (1U << 1)
#define IORING_POLL_UPDATE_USER_DATA (1U << 2)
#define IORING_POLL_ADD_LEVEL (1U << 3)
/*
* ASYNC_CANCEL flags.
*
* IORING_ASYNC_CANCEL_ALL Cancel all requests that match the given key
* IORING_ASYNC_CANCEL_FD Key off 'fd' for cancelation rather than the
* request 'user_data'
* IORING_ASYNC_CANCEL_ANY Match any request
* IORING_ASYNC_CANCEL_FD_FIXED 'fd' passed in is a fixed descriptor
* IORING_ASYNC_CANCEL_USERDATA Match on user_data, default for no other key
* IORING_ASYNC_CANCEL_OP Match request based on opcode
*/
#define IORING_ASYNC_CANCEL_ALL (1U << 0)
#define IORING_ASYNC_CANCEL_FD (1U << 1)
#define IORING_ASYNC_CANCEL_ANY (1U << 2)
#define IORING_ASYNC_CANCEL_FD_FIXED (1U << 3)
#define IORING_ASYNC_CANCEL_USERDATA (1U << 4)
#define IORING_ASYNC_CANCEL_OP (1U << 5)
/*
* send/sendmsg and recv/recvmsg flags (sqe->ioprio)
*
* IORING_RECVSEND_POLL_FIRST If set, instead of first attempting to send
* or receive and arm poll if that yields an
* -EAGAIN result, arm poll upfront and skip
* the initial transfer attempt.
*
* IORING_RECV_MULTISHOT Multishot recv. Sets IORING_CQE_F_MORE if
* the handler will continue to report
* CQEs on behalf of the same SQE.
*
* IORING_RECVSEND_FIXED_BUF Use registered buffers, the index is stored in
* the buf_index field.
*
* IORING_SEND_ZC_REPORT_USAGE
* If set, SEND[MSG]_ZC should report
* the zerocopy usage in cqe.res
* for the IORING_CQE_F_NOTIF cqe.
* 0 is reported if zerocopy was actually possible.
* IORING_NOTIF_USAGE_ZC_COPIED if data was copied
* (at least partially).
*
* IORING_RECVSEND_BUNDLE Used with IOSQE_BUFFER_SELECT. If set, send or
* recv will grab as many buffers from the buffer
* group ID given and send them all. The completion
* result will be the number of buffers send, with
* the starting buffer ID in cqe->flags as per
* usual for provided buffer usage. The buffers
* will be contiguous from the starting buffer ID.
*
* IORING_SEND_VECTORIZED If set, SEND[_ZC] will take a pointer to a io_vec
* to allow vectorized send operations.
*/
#define IORING_RECVSEND_POLL_FIRST (1U << 0)
#define IORING_RECV_MULTISHOT (1U << 1)
#define IORING_RECVSEND_FIXED_BUF (1U << 2)
#define IORING_SEND_ZC_REPORT_USAGE (1U << 3)
#define IORING_RECVSEND_BUNDLE (1U << 4)
#define IORING_SEND_VECTORIZED (1U << 5)
/*
* cqe.res for IORING_CQE_F_NOTIF if
* IORING_SEND_ZC_REPORT_USAGE was requested
*
* It should be treated as a flag, all other
* bits of cqe.res should be treated as reserved!
*/
#define IORING_NOTIF_USAGE_ZC_COPIED (1U << 31)
/*
* accept flags stored in sqe->ioprio
*/
#define IORING_ACCEPT_MULTISHOT (1U << 0)
#define IORING_ACCEPT_DONTWAIT (1U << 1)
#define IORING_ACCEPT_POLL_FIRST (1U << 2)
/*
* IORING_OP_MSG_RING command types, stored in sqe->addr
*/
enum io_uring_msg_ring_flags {
IORING_MSG_DATA,
/* pass sqe->len as 'res' and off as user_data */
IORING_MSG_SEND_FD,
/* send a registered fd to another ring */
};
/*
* IORING_OP_MSG_RING flags (sqe->msg_ring_flags)
*
* IORING_MSG_RING_CQE_SKIP Don't post a CQE to the target ring. Not
* applicable for IORING_MSG_DATA, obviously.
*/
#define IORING_MSG_RING_CQE_SKIP (1U << 0)
/* Pass through the flags from sqe->file_index to cqe->flags */
#define IORING_MSG_RING_FLAGS_PASS (1U << 1)
/*
* IORING_OP_FIXED_FD_INSTALL flags (sqe->install_fd_flags)
*
* IORING_FIXED_FD_NO_CLOEXEC Don't mark the fd as O_CLOEXEC
*/
#define IORING_FIXED_FD_NO_CLOEXEC (1U << 0)
/*
* IORING_OP_NOP flags (sqe->nop_flags)
*
* IORING_NOP_INJECT_RESULT Inject result from sqe->result
*/
#define IORING_NOP_INJECT_RESULT (1U << 0)
#define IORING_NOP_CQE32 (1U << 5)
/*
* IO completion data structure (Completion Queue Entry)
*/
struct io_uring_cqe {
__u64 user_data;
/* sqe->user_data value passed back */
__s32 res;
/* result code for this event */
__u32 flags;
/*
* If the ring is initialized with IORING_SETUP_CQE32, then this field
* contains 16-bytes of padding, doubling the size of the CQE.
*/
__u64 big_cqe[];
};
/*
* cqe->flags
*
* IORING_CQE_F_BUFFER If set, the upper 16 bits are the buffer ID
* IORING_CQE_F_MORE If set, parent SQE will generate more CQE entries
* IORING_CQE_F_SOCK_NONEMPTY If set, more data to read after socket recv
* IORING_CQE_F_NOTIF Set for notification CQEs. Can be used to distinct
* them from sends.
* IORING_CQE_F_BUF_MORE If set, the buffer ID set in the completion will get
* more completions. In other words, the buffer is being
* partially consumed, and will be used by the kernel for
* more completions. This is only set for buffers used via
* the incremental buffer consumption, as provided by
* a ring buffer setup with IOU_PBUF_RING_INC. For any
* other provided buffer type, all completions with a
* buffer passed back is automatically returned to the
* application.
* IORING_CQE_F_SKIP If set, then the application/liburing must ignore this
* CQE. It's only purpose is to fill a gap in the ring,
* if a large CQE is attempted posted when the ring has
* just a single small CQE worth of space left before
* wrapping.
* IORING_CQE_F_32 If set, this is a 32b/big-cqe posting. Use with rings
* setup in a mixed CQE mode, where both 16b and 32b
* CQEs may be posted to the CQ ring.
*/
#define IORING_CQE_F_BUFFER (1U << 0)
#define IORING_CQE_F_MORE (1U << 1)
#define IORING_CQE_F_SOCK_NONEMPTY (1U << 2)
#define IORING_CQE_F_NOTIF (1U << 3)
#define IORING_CQE_F_BUF_MORE (1U << 4)
#define IORING_CQE_F_SKIP (1U << 5)
#define IORING_CQE_F_32 (1U << 15)
#define IORING_CQE_BUFFER_SHIFT 16
/*
* Magic offsets for the application to mmap the data it needs
*/
#define IORING_OFF_SQ_RING 0ULL
#define IORING_OFF_CQ_RING 0x8000000ULL
#define IORING_OFF_SQES 0x10000000ULL
#define IORING_OFF_PBUF_RING 0x80000000ULL
#define IORING_OFF_PBUF_SHIFT 16
#define IORING_OFF_MMAP_MASK 0xf8000000ULL
/*
* Filled with the offset for mmap(2)
*/
struct io_sqring_offsets {
__u32 head;
__u32 tail;
__u32 ring_mask;
__u32 ring_entries;
__u32 flags;
__u32 dropped;
__u32 array;
__u32 resv1;
__u64 user_addr;
};
/*
* sq_ring->flags
*/
#define IORING_SQ_NEED_WAKEUP (1U << 0)
/* needs io_uring_enter wakeup */
#define IORING_SQ_CQ_OVERFLOW (1U << 1)
/* CQ ring is overflown */
#define IORING_SQ_TASKRUN (1U << 2)
/* task should enter the kernel */
struct io_cqring_offsets {
__u32 head;
__u32 tail;
__u32 ring_mask;
__u32 ring_entries;
__u32 overflow;
__u32 cqes;
__u32 flags;
__u32 resv1;
__u64 user_addr;
};
/*
* cq_ring->flags
*/
/* disable eventfd notifications */
#define IORING_CQ_EVENTFD_DISABLED (1U << 0)
/*
* io_uring_enter(2) flags
*/
#define IORING_ENTER_GETEVENTS (1U << 0)
#define IORING_ENTER_SQ_WAKEUP (1U << 1)
#define IORING_ENTER_SQ_WAIT (1U << 2)
#define IORING_ENTER_EXT_ARG (1U << 3)
#define IORING_ENTER_REGISTERED_RING (1U << 4)
#define IORING_ENTER_ABS_TIMER (1U << 5)
#define IORING_ENTER_EXT_ARG_REG (1U << 6)
#define IORING_ENTER_NO_IOWAIT (1U << 7)
/*
* Passed in for io_uring_setup(2). Copied back with updated info on success
*/
struct io_uring_params {
__u32 sq_entries;
__u32 cq_entries;
__u32 flags;
__u32 sq_thread_cpu;
__u32 sq_thread_idle;
__u32 features;
__u32 wq_fd;
__u32 resv[3];
struct io_sqring_offsets sq_off;
struct io_cqring_offsets cq_off;
};
/*
* io_uring_params->features flags
*/
#define IORING_FEAT_SINGLE_MMAP (1U << 0)
#define IORING_FEAT_NODROP (1U << 1)
#define IORING_FEAT_SUBMIT_STABLE (1U << 2)
#define IORING_FEAT_RW_CUR_POS (1U << 3)
#define IORING_FEAT_CUR_PERSONALITY (1U << 4)
#define IORING_FEAT_FAST_POLL (1U << 5)
#define IORING_FEAT_POLL_32BITS (1U << 6)
#define IORING_FEAT_SQPOLL_NONFIXED (1U << 7)
#define IORING_FEAT_EXT_ARG (1U << 8)
#define IORING_FEAT_NATIVE_WORKERS (1U << 9)
#define IORING_FEAT_RSRC_TAGS (1U << 10)
#define IORING_FEAT_CQE_SKIP (1U << 11)
#define IORING_FEAT_LINKED_FILE (1U << 12)
#define IORING_FEAT_REG_REG_RING (1U << 13)
#define IORING_FEAT_RECVSEND_BUNDLE (1U << 14)
#define IORING_FEAT_MIN_TIMEOUT (1U << 15)
#define IORING_FEAT_RW_ATTR (1U << 16)
#define IORING_FEAT_NO_IOWAIT (1U << 17)
/*
* io_uring_register(2) opcodes and arguments
*/
enum io_uring_register_op {
IORING_REGISTER_BUFFERS = 0,
IORING_UNREGISTER_BUFFERS = 1,
IORING_REGISTER_FILES = 2,
IORING_UNREGISTER_FILES = 3,
IORING_REGISTER_EVENTFD = 4,
IORING_UNREGISTER_EVENTFD = 5,
IORING_REGISTER_FILES_UPDATE = 6,
IORING_REGISTER_EVENTFD_ASYNC = 7,
IORING_REGISTER_PROBE = 8,
IORING_REGISTER_PERSONALITY = 9,
IORING_UNREGISTER_PERSONALITY = 10,
IORING_REGISTER_RESTRICTIONS = 11,
IORING_REGISTER_ENABLE_RINGS = 12,
/* extended with tagging */
IORING_REGISTER_FILES2 = 13,
IORING_REGISTER_FILES_UPDATE2 = 14,
IORING_REGISTER_BUFFERS2 = 15,
IORING_REGISTER_BUFFERS_UPDATE = 16,
/* set/clear io-wq thread affinities */
IORING_REGISTER_IOWQ_AFF = 17,
IORING_UNREGISTER_IOWQ_AFF = 18,
/* set/get max number of io-wq workers */
IORING_REGISTER_IOWQ_MAX_WORKERS = 19,
/* register/unregister io_uring fd with the ring */
IORING_REGISTER_RING_FDS = 20,
IORING_UNREGISTER_RING_FDS = 21,
/* register ring based provide buffer group */
IORING_REGISTER_PBUF_RING = 22,
IORING_UNREGISTER_PBUF_RING = 23,
/* sync cancelation API */
IORING_REGISTER_SYNC_CANCEL = 24,
/* register a range of fixed file slots for automatic slot allocation */
IORING_REGISTER_FILE_ALLOC_RANGE = 25,
/* return status information for a buffer group */
IORING_REGISTER_PBUF_STATUS = 26,
/* set/clear busy poll settings */
IORING_REGISTER_NAPI = 27,
IORING_UNREGISTER_NAPI = 28,
IORING_REGISTER_CLOCK = 29,
/* clone registered buffers from source ring to current ring */
IORING_REGISTER_CLONE_BUFFERS = 30,
/* send MSG_RING without having a ring */
IORING_REGISTER_SEND_MSG_RING = 31,
/* register a netdev hw rx queue for zerocopy */
IORING_REGISTER_ZCRX_IFQ = 32,
/* resize CQ ring */
IORING_REGISTER_RESIZE_RINGS = 33,
IORING_REGISTER_MEM_REGION = 34,
/* query various aspects of io_uring, see linux/io_uring/query.h */
IORING_REGISTER_QUERY = 35,
/* auxiliary zcrx configuration, see enum zcrx_ctrl_op */
IORING_REGISTER_ZCRX_CTRL = 36,
/* register bpf filtering programs */
IORING_REGISTER_BPF_FILTER = 37,
/* this goes last */
IORING_REGISTER_LAST,
/* flag added to the opcode to use a registered ring fd */
IORING_REGISTER_USE_REGISTERED_RING = 1U << 31
};
/* io-wq worker categories */
enum io_wq_type {
IO_WQ_BOUND,
IO_WQ_UNBOUND,
};
/* deprecated, see struct io_uring_rsrc_update */
struct io_uring_files_update {
__u32 offset;
__u32 resv;
__aligned_u64
/* __s32 * */ fds;
};
enum {
/* initialise with user provided memory pointed by user_addr */
IORING_MEM_REGION_TYPE_USER = 1,
};
struct io_uring_region_desc {
__u64 user_addr;
__u64 size;
__u32 flags;
__u32 id;
__u64 mmap_offset;
__u64 __resv[4];
};
enum {
/* expose the region as registered wait arguments */
IORING_MEM_REGION_REG_WAIT_ARG = 1,
};
struct io_uring_mem_region_reg {
__u64 region_uptr;
/* struct io_uring_region_desc * */
__u64 flags;
__u64 __resv[2];
};
/*
* Register a fully sparse file space, rather than pass in an array of all
* -1 file descriptors.
*/
#define IORING_RSRC_REGISTER_SPARSE (1U << 0)
struct io_uring_rsrc_register {
__u32 nr;
__u32 flags;
__u64 resv2;
__aligned_u64 data;
__aligned_u64 tags;
};
struct io_uring_rsrc_update {
__u32 offset;
__u32 resv;
__aligned_u64 data;
};
struct io_uring_rsrc_update2 {
__u32 offset;
__u32 resv;
__aligned_u64 data;
__aligned_u64 tags;
__u32 nr;
__u32 resv2;
};
/* Skip updating fd indexes set to this value in the fd table */
#define IORING_REGISTER_FILES_SKIP (-2)
#define IO_URING_OP_SUPPORTED (1U << 0)
struct io_uring_probe_op {
__u8 op;
__u8 resv;
__u16 flags;
/* IO_URING_OP_* flags */
__u32 resv2;
};
struct io_uring_probe {
__u8 last_op;
/* last opcode supported */
__u8 ops_len;
/* length of ops[] array below */
__u16 resv;
__u32 resv2[3];
struct io_uring_probe_op ops[];
};
struct io_uring_restriction {
__u16 opcode;
union {
__u8 register_op;
/* IORING_RESTRICTION_REGISTER_OP */
__u8 sqe_op;
/* IORING_RESTRICTION_SQE_OP */
__u8 sqe_flags;
/* IORING_RESTRICTION_SQE_FLAGS_* */
};
__u8 resv;
__u32 resv2[3];
};
struct io_uring_task_restriction {
__u16 flags;
__u16 nr_res;
__u32 resv[3];
struct io_uring_restriction restrictions[0];
};
struct io_uring_clock_register {
__u32 clockid;
__u32 __resv[3];
};
enum {
IORING_REGISTER_SRC_REGISTERED = (1U << 0),
IORING_REGISTER_DST_REPLACE = (1U << 1),
};
struct io_uring_clone_buffers {
__u32 src_fd;
__u32 flags;
__u32 src_off;
__u32 dst_off;
__u32 nr;
__u32 pad[3];
};
struct io_uring_buf {
__u64 addr;
__u32 len;
__u16 bid;
__u16 resv;
};
struct io_uring_buf_ring {
union {
/*
* To avoid spilling into more pages than we need to, the
* ring tail is overlaid with the io_uring_buf->resv field.
*/
struct {
__u64 resv1;
__u32 resv2;
__u16 resv3;
__u16 tail;
};
struct io_uring_buf bufs[0];
};
};
/*
* Flags for IORING_REGISTER_PBUF_RING.
*
* IOU_PBUF_RING_MMAP: If set, kernel will allocate the memory for the ring.
* The application must not set a ring_addr in struct
* io_uring_buf_reg, instead it must subsequently call
* mmap(2) with the offset set as:
* IORING_OFF_PBUF_RING | (bgid << IORING_OFF_PBUF_SHIFT)
* to get a virtual mapping for the ring.
* IOU_PBUF_RING_INC: If set, buffers consumed from this buffer ring can be
* consumed incrementally. Normally one (or more) buffers
* are fully consumed. With incremental consumptions, it's
* feasible to register big ranges of buffers, and each
* use of it will consume only as much as it needs. This
* requires that both the kernel and application keep
* track of where the current read/recv index is at.
*/
enum io_uring_register_pbuf_ring_flags {
IOU_PBUF_RING_MMAP = 1,
IOU_PBUF_RING_INC = 2,
};
/* argument for IORING_(UN)REGISTER_PBUF_RING */
struct io_uring_buf_reg {
__u64 ring_addr;
__u32 ring_entries;
__u16 bgid;
__u16 flags;
__u32 min_left;
__u32 resv[5];
};
/* argument for IORING_REGISTER_PBUF_STATUS */
struct io_uring_buf_status {
__u32 buf_group;
/* input */
__u32 head;
/* output */
__u32 resv[8];
};
/* argument for IORING_(UN)REGISTER_NAPI */
struct io_uring_napi {
__u32 busy_poll_to;
__u8 prefer_busy_poll;
__u8 pad[3];
__u64 resv;
};
/*
* io_uring_restriction->opcode values
*/
enum io_uring_register_restriction_op {
/* Allow an io_uring_register(2) opcode */
IORING_RESTRICTION_REGISTER_OP = 0,
/* Allow an sqe opcode */
IORING_RESTRICTION_SQE_OP = 1,
/* Allow sqe flags */
IORING_RESTRICTION_SQE_FLAGS_ALLOWED = 2,
/* Require sqe flags (these flags must be set on each submission) */
IORING_RESTRICTION_SQE_FLAGS_REQUIRED = 3,
IORING_RESTRICTION_LAST
};
enum {
IORING_REG_WAIT_TS = (1U << 0),
};
/*
* Argument for io_uring_enter(2) with
* IORING_GETEVENTS | IORING_ENTER_EXT_ARG_REG set, where the actual argument
* is an index into a previously registered fixed wait region described by
* the below structure.
*/
struct io_uring_reg_wait {
struct __kernel_timespec ts;
__u32 min_wait_usec;
__u32 flags;
__u64 sigmask;
__u32 sigmask_sz;
__u32 pad[3];
__u64 pad2[2];
};
/*
* Argument for io_uring_enter(2) with IORING_GETEVENTS | IORING_ENTER_EXT_ARG
*/
struct io_uring_getevents_arg {
__u64 sigmask;
__u32 sigmask_sz;
__u32 min_wait_usec;
__u64 ts;
};
/*
* Argument for IORING_REGISTER_SYNC_CANCEL
*/
struct io_uring_sync_cancel_reg {
__u64 addr;
__s32 fd;
__u32 flags;
struct __kernel_timespec timeout;
__u8 opcode;
__u8 pad[7];
__u64 pad2[3];
};
/*
* Argument for IORING_REGISTER_FILE_ALLOC_RANGE
* The range is specified as [off, off + len)
*/
struct io_uring_file_index_range {
__u32 off;
__u32 len;
__u64 resv;
};
struct io_uring_recvmsg_out {
__u32 namelen;
__u32 controllen;
__u32 payloadlen;
__u32 flags;
};
/*
* Argument for IORING_OP_URING_CMD when file is a socket
*/
enum io_uring_socket_op {
SOCKET_URING_OP_SIOCINQ = 0,
SOCKET_URING_OP_SIOCOUTQ,
SOCKET_URING_OP_GETSOCKOPT,
SOCKET_URING_OP_SETSOCKOPT,
SOCKET_URING_OP_TX_TIMESTAMP,
SOCKET_URING_OP_GETSOCKNAME,
};
/*
* SOCKET_URING_OP_TX_TIMESTAMP definitions
*/
#define IORING_TIMESTAMP_HW_SHIFT 16
/* The cqe->flags bit from which the timestamp type is stored */
#define IORING_TIMESTAMP_TYPE_SHIFT (IORING_TIMESTAMP_HW_SHIFT + 1)
/* The cqe->flags flag signifying whether it's a hardware timestamp */
#define IORING_CQE_F_TSTAMP_HW ((__u32)1 << IORING_TIMESTAMP_HW_SHIFT)
struct io_timespec {
__u64 tv_sec;
__u64 tv_nsec;
};
/* Zero copy receive refill queue entry */
struct io_uring_zcrx_rqe {
__u64 off;
__u32 len;
__u32 __pad;
};
struct io_uring_zcrx_cqe {
__u64 off;
__u64 __pad;
};
/* The bit from which area id is encoded into offsets */
#define IORING_ZCRX_AREA_SHIFT 48
#define IORING_ZCRX_AREA_MASK (~(((__u64)1 << IORING_ZCRX_AREA_SHIFT) - 1))
struct io_uring_zcrx_offsets {
__u32 head;
__u32 tail;
__u32 rqes;
__u32 __resv2;
__u64 __resv[2];
};
enum io_uring_zcrx_area_flags {
IORING_ZCRX_AREA_DMABUF = 1,
};
struct io_uring_zcrx_area_reg {
__u64 addr;
__u64 len;
__u64 rq_area_token;
__u32 flags;
__u32 dmabuf_fd;
__u64 __resv2[2];
};
enum zcrx_reg_flags {
ZCRX_REG_IMPORT = 1,
/*
* Register a zcrx instance without a net device. All data will be
* copied. The refill queue entries might not be automatically
* consumed and need to be flushed, see ZCRX_CTRL_FLUSH_RQ.
*/
ZCRX_REG_NODEV = 2,
};
enum zcrx_features {
/*
* The user can ask for the desired rx page size by passing the
* value in struct io_uring_zcrx_ifq_reg::rx_buf_len.
*/
ZCRX_FEATURE_RX_PAGE_SIZE = 1 << 0,
};
/*
* Argument for IORING_REGISTER_ZCRX_IFQ
*/
struct io_uring_zcrx_ifq_reg {
__u32 if_idx;
__u32 if_rxq;
__u32 rq_entries;
__u32 flags;
__u64 area_ptr;
/* pointer to struct io_uring_zcrx_area_reg */
__u64 region_ptr;
/* struct io_uring_region_desc * */
struct io_uring_zcrx_offsets offsets;
__u32 zcrx_id;
__u32 rx_buf_len;
__u64 __resv[3];
};
enum zcrx_ctrl_op {
ZCRX_CTRL_FLUSH_RQ,
ZCRX_CTRL_EXPORT,
__ZCRX_CTRL_LAST,
};
struct zcrx_ctrl_flush_rq {
__u64 __resv[6];
};
struct zcrx_ctrl_export {
__u32 zcrx_fd;
__u32 __resv1[11];
};
struct zcrx_ctrl {
__u32 zcrx_id;
__u32 op;
/* see enum zcrx_ctrl_op */
__u64 __resv[2];
union {
struct zcrx_ctrl_export zc_export;
struct zcrx_ctrl_flush_rq zc_flush;
};
};
#ifdef __cplusplus
}
#endif
#endif
/* SPDX-License-Identifier: (GPL-2.0 WITH Linux-syscall-note) OR MIT */
/*
* Header file for the io_uring BPF filters.
*/
#ifndef LINUX_IO_URING_BPF_FILTER_H
#define LINUX_IO_URING_BPF_FILTER_H
#include <linux/types.h>
/*
* Struct passed to filters.
*/
struct io_uring_bpf_ctx {
__u64 user_data;
__u8 opcode;
__u8 sqe_flags;
__u8 pdu_size;
/* size of aux data for filter */
__u8 pad[5];
union {
struct {
__u32 family;
__u32 type;
__u32 protocol;
} socket;
struct {
__u64 flags;
__u64 mode;
__u64 resolve;
} open;
};
};
enum {
/*
* If set, any currently unset opcode will have a deny filter attached
*/
IO_URING_BPF_FILTER_DENY_REST = 1,
/*
* If set, if kernel and application don't agree on pdu_size for
* the given opcode, fail the registration of the filter.
*/
IO_URING_BPF_FILTER_SZ_STRICT = 2,
};
struct io_uring_bpf_filter {
__u32 opcode;
/* io_uring opcode to filter */
__u32 flags;
__u32 filter_len;
/* number of BPF instructions */
__u8 pdu_size;
/* expected pdu size for opcode */
__u8 resv[3];
__u64 filter_ptr;
/* pointer to BPF filter */
__u64 resv2[5];
};
enum {
IO_URING_BPF_CMD_FILTER = 1,
};
struct io_uring_bpf {
__u16 cmd_type;
/* IO_URING_BPF_* values */
__u16 cmd_flags;
/* none so far */
__u32 resv;
union {
struct io_uring_bpf_filter filter;
};
};
#endif
/* SPDX-License-Identifier: (GPL-2.0 WITH Linux-syscall-note) OR MIT */
/*
* Header file for the io_uring query interface.
*/
#ifndef LINUX_IO_URING_QUERY_H
#define LINUX_IO_URING_QUERY_H
#include <linux/types.h>
struct io_uring_query_hdr {
__u64 next_entry;
__u64 query_data;
__u32 query_op;
__u32 size;
__s32 result;
__u32 __resv[3];
};
enum {
IO_URING_QUERY_OPCODES = 0,
IO_URING_QUERY_ZCRX = 1,
IO_URING_QUERY_SCQ = 2,
__IO_URING_QUERY_MAX,
};
/* Doesn't require a ring */
struct io_uring_query_opcode {
/* The number of supported IORING_OP_* opcodes */
__u32 nr_request_opcodes;
/* The number of supported IORING_[UN]REGISTER_* opcodes */
__u32 nr_register_opcodes;
/* Bitmask of all supported IORING_FEAT_* flags */
__u64 feature_flags;
/* Bitmask of all supported IORING_SETUP_* flags */
__u64 ring_setup_flags;
/* Bitmask of all supported IORING_ENTER_** flags */
__u64 enter_flags;
/* Bitmask of all supported IOSQE_* flags */
__u64 sqe_flags;
/* The number of available query opcodes */
__u32 nr_query_opcodes;
__u32 __pad;
};
struct io_uring_query_zcrx {
/* Bitmask of supported ZCRX_REG_* flags, */
__u64 register_flags;
/* Bitmask of all supported IORING_ZCRX_AREA_* flags */
__u64 area_flags;
/* The number of supported ZCRX_CTRL_* opcodes */
__u32 nr_ctrl_opcodes;
/* Bitmask of ZCRX_FEATURE_* indicating which features are available */
__u32 features;
/* The refill ring header size */
__u32 rq_hdr_size;
/* The alignment for the header */
__u32 rq_hdr_alignment;
__u64 __resv2;
};
struct io_uring_query_scq {
/* The SQ/CQ rings header size */
__u64 hdr_size;
/* The alignment for the header */
__u64 hdr_alignment;
};
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_SANITIZE_H
#define LIBURING_SANITIZE_H
#ifdef __cplusplus
#define LIBURING_NOEXCEPT noexcept
#else
#define LIBURING_NOEXCEPT
#endif
#ifdef __cplusplus
extern "C" {
#endif
struct io_uring;
struct iovec;
#if defined(CONFIG_USE_SANITIZER)
void liburing_sanitize_ring(struct io_uring *ring) LIBURING_NOEXCEPT;
void liburing_sanitize_address(const void *addr) LIBURING_NOEXCEPT;
void liburing_sanitize_region(const void *addr, unsigned int len)
LIBURING_NOEXCEPT;
void liburing_sanitize_iovecs(const struct iovec *iovecs, unsigned nr)
LIBURING_NOEXCEPT;
#else
#define __maybe_unused __attribute__((__unused__))
static inline void liburing_sanitize_ring(struct io_uring __maybe_unused *ring)
LIBURING_NOEXCEPT{
}
static inline void liburing_sanitize_address(const void __maybe_unused *addr)
LIBURING_NOEXCEPT{
}
static inline void liburing_sanitize_region(const void __maybe_unused *addr,
unsigned int __maybe_unused len)
LIBURING_NOEXCEPT{
}
static inline void liburing_sanitize_iovecs(const struct iovec __maybe_unused *iovecs,
unsigned __maybe_unused nr)
LIBURING_NOEXCEPT{
}
#endif
#ifdef __cplusplus
}
#endif
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIB_URING_H
#define LIB_URING_H
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>
#include <fcntl.h>
#include <sched.h>
#include <linux/swab.h>
#include <linux/filter.h>
#include <sys/wait.h>
#include "liburing/compat.h"
#include "liburing/io_uring.h"
#include "liburing/io_uring/query.h"
#include "liburing/io_uring/bpf_filter.h"
#include "liburing/io_uring_version.h"
#ifndef uring_unlikely
#define uring_unlikely(cond) __builtin_expect(!!(cond), 0)
#endif
#ifndef uring_likely
#define uring_likely(cond) __builtin_expect(!!(cond), 1)
#endif
/*
* NOTE: Use IOURINGINLINE macro for "static inline" functions that are
* expected to be available in the FFI bindings. They must also
* be included in the liburing-ffi.map file.
*
* Use _LOCAL_INLINE macro for "static inline" functions that are
* not expected to be available in the FFI bindings.
*
* Don't use "static inline" directly when defining new functions
* in this header file.
*
* Reason:
* The C++20 module export feature fails to operate correctly
* with the "static inline" functions. Use "inline" instead of
* "static inline" when compiling with C++20 or later.
*
* See:
* https:
//github.com/axboe/liburing/issues/1457
* https:
//lore.kernel.org/io-uring/e0559c10-104d-4da8-9f7f-d2ffd73d8df3@acm.org
*/
#ifndef IOURINGINLINE
#if defined(__cplusplus) && __cplusplus >= 202002L
#define IOURINGINLINE inline
#else
#define IOURINGINLINE static inline
#endif
#endif
#ifndef _LOCAL_INLINE
#if defined(__cplusplus) && __cplusplus >= 202002L
#define _LOCAL_INLINE inline
#else
#define _LOCAL_INLINE static inline
#endif
#endif
/*
* barrier.h needs _LOCAL_INLINE.
*/
#include "liburing/barrier.h"
#ifdef __alpha__
/*
* alpha and mips are the exceptions, all other architectures have
* common numbers for new system calls.
*/
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 535
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 536
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register 537
#endif
#elif defined __mips__
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup (__NR_Linux + 425)
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter (__NR_Linux + 426)
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register (__NR_Linux + 427)
#endif
#else
/* !__alpha__ and !__mips__ */
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif
#endif
#ifdef __cplusplus
#define LIBURING_NOEXCEPT noexcept
#else
#define LIBURING_NOEXCEPT
#endif
#ifdef __cplusplus
extern "C" {
#endif
/*
* Library interface to io_uring
*/
struct io_uring_sq {
unsigned *khead;
unsigned *ktail;
// Deprecated: use `ring_mask` instead of `*kring_mask`
unsigned *kring_mask;
// Deprecated: use `ring_entries` instead of `*kring_entries`
unsigned *kring_entries;
unsigned *kflags;
unsigned *kdropped;
unsigned *array;
struct io_uring_sqe *sqes;
unsigned sqe_head;
unsigned sqe_tail;
size_t ring_sz;
void *ring_ptr;
unsigned ring_mask;
unsigned ring_entries;
unsigned sqes_sz;
unsigned pad;
};
struct io_uring_cq {
unsigned *khead;
unsigned *ktail;
// Deprecated: use `ring_mask` instead of `*kring_mask`
unsigned *kring_mask;
// Deprecated: use `ring_entries` instead of `*kring_entries`
unsigned *kring_entries;
unsigned *kflags;
unsigned *koverflow;
struct io_uring_cqe *cqes;
size_t ring_sz;
void *ring_ptr;
unsigned ring_mask;
unsigned ring_entries;
unsigned pad[2];
};
struct io_uring {
struct io_uring_sq sq;
struct io_uring_cq cq;
unsigned flags;
int ring_fd;
unsigned features;
int enter_ring_fd;
__u8 int_flags;
__u8 pad[3];
unsigned pad2;
};
struct io_uring_zcrx_rq {
__u32 *khead;
__u32 *ktail;
__u32 rq_tail;
unsigned ring_entries;
struct io_uring_zcrx_rqe *rqes;
void *ring_ptr;
};
/*
* Library interface
*/
_LOCAL_INLINE __u64 uring_ptr_to_u64(const void *ptr) LIBURING_NOEXCEPT{
return (__u64) (unsigned long) ptr;
}
/*
* return an allocated io_uring_probe structure, or NULL if probe fails (for
* example, if it is not available). The caller is responsible for freeing it
*/
struct io_uring_probe *io_uring_get_probe_ring(struct io_uring *ring)
LIBURING_NOEXCEPT;
/* same as io_uring_get_probe_ring, but takes care of ring init and teardown */
struct io_uring_probe *io_uring_get_probe(void) LIBURING_NOEXCEPT;
/*
* frees a probe allocated through io_uring_get_probe() or
* io_uring_get_probe_ring()
*/
void io_uring_free_probe(struct io_uring_probe *probe) LIBURING_NOEXCEPT;
IOURINGINLINE int io_uring_opcode_supported(const struct io_uring_probe *p,
int op) LIBURING_NOEXCEPT{
if (op < 0 || op > p->last_op)
return 0;
return (p->ops[op].flags & IO_URING_OP_SUPPORTED) != 0;
}
int io_uring_queue_init_mem(unsigned entries, struct io_uring *ring,
struct io_uring_params *p,
void *buf, size_t buf_size) LIBURING_NOEXCEPT;
int io_uring_queue_init_params(unsigned entries, struct io_uring *ring,
struct io_uring_params *p) LIBURING_NOEXCEPT;
int io_uring_queue_init(unsigned entries, struct io_uring *ring,
unsigned flags) LIBURING_NOEXCEPT;
int io_uring_queue_mmap(int fd, struct io_uring_params *p,
struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_ring_dontfork(struct io_uring *ring) LIBURING_NOEXCEPT;
void io_uring_queue_exit(struct io_uring *ring) LIBURING_NOEXCEPT;
unsigned io_uring_peek_batch_cqe(struct io_uring *ring,
struct io_uring_cqe **cqes, unsigned count) LIBURING_NOEXCEPT;
int io_uring_wait_cqes(struct io_uring *ring, struct io_uring_cqe **cqe_ptr,
unsigned wait_nr, struct __kernel_timespec *ts,
sigset_t *sigmask) LIBURING_NOEXCEPT;
int io_uring_wait_cqes_min_timeout(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr,
unsigned wait_nr,
struct __kernel_timespec *ts,
unsigned int min_ts_usec,
sigset_t *sigmask) LIBURING_NOEXCEPT;
int io_uring_wait_cqe_timeout(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr,
struct __kernel_timespec *ts) LIBURING_NOEXCEPT;
int io_uring_submit(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_submit_and_wait(struct io_uring *ring, unsigned wait_nr)
LIBURING_NOEXCEPT;
int io_uring_submit_and_wait_timeout(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr,
unsigned wait_nr,
struct __kernel_timespec *ts,
sigset_t *sigmask) LIBURING_NOEXCEPT;
int io_uring_submit_and_wait_min_timeout(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr,
unsigned wait_nr,
struct __kernel_timespec *ts,
unsigned min_wait,
sigset_t *sigmask) LIBURING_NOEXCEPT;
int io_uring_submit_and_wait_reg(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr, unsigned wait_nr,
int reg_index) LIBURING_NOEXCEPT;
int io_uring_register_wait_reg(struct io_uring *ring,
struct io_uring_reg_wait *reg, int nr)
LIBURING_NOEXCEPT;
int io_uring_resize_rings(struct io_uring *ring, struct io_uring_params *p)
LIBURING_NOEXCEPT;
int io_uring_clone_buffers_offset(struct io_uring *dst, struct io_uring *src,
unsigned int dst_off, unsigned int src_off,
unsigned int nr, unsigned int flags)
LIBURING_NOEXCEPT;
int __io_uring_clone_buffers_offset(struct io_uring *dst, struct io_uring *src,
unsigned int dst_off, unsigned int src_off,
unsigned int nr, unsigned int flags)
LIBURING_NOEXCEPT;
int io_uring_clone_buffers(struct io_uring *dst, struct io_uring *src)
LIBURING_NOEXCEPT;
int __io_uring_clone_buffers(struct io_uring *dst, struct io_uring *src,
unsigned int flags) LIBURING_NOEXCEPT;
int io_uring_register_buffers(struct io_uring *ring, const struct iovec *iovecs,
unsigned nr_iovecs) LIBURING_NOEXCEPT;
int io_uring_register_buffers_tags(struct io_uring *ring,
const struct iovec *iovecs,
const __u64 *tags, unsigned nr)
LIBURING_NOEXCEPT;
int io_uring_register_buffers_sparse(struct io_uring *ring, unsigned nr)
LIBURING_NOEXCEPT;
int io_uring_register_buffers_update_tag(struct io_uring *ring,
unsigned off,
const struct iovec *iovecs,
const __u64 *tags, unsigned nr)
LIBURING_NOEXCEPT;
int io_uring_unregister_buffers(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_register_files(struct io_uring *ring, const int *files,
unsigned nr_files) LIBURING_NOEXCEPT;
int io_uring_register_files_tags(struct io_uring *ring, const int *files,
const __u64 *tags, unsigned nr)
LIBURING_NOEXCEPT;
int io_uring_register_files_sparse(struct io_uring *ring, unsigned nr)
LIBURING_NOEXCEPT;
int io_uring_register_files_update_tag(struct io_uring *ring, unsigned off,
const int *files, const __u64 *tags,
unsigned nr_files) LIBURING_NOEXCEPT;
int io_uring_unregister_files(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_register_files_update(struct io_uring *ring, unsigned off,
const int *files, unsigned nr_files)
LIBURING_NOEXCEPT;
int io_uring_register_eventfd(struct io_uring *ring, int fd) LIBURING_NOEXCEPT;
int io_uring_register_eventfd_async(struct io_uring *ring, int fd)
LIBURING_NOEXCEPT;
int io_uring_unregister_eventfd(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_register_probe(struct io_uring *ring, struct io_uring_probe *p,
unsigned nr) LIBURING_NOEXCEPT;
int io_uring_register_personality(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_unregister_personality(struct io_uring *ring, int id)
LIBURING_NOEXCEPT;
int io_uring_register_restrictions(struct io_uring *ring,
struct io_uring_restriction *res,
unsigned int nr_res) LIBURING_NOEXCEPT;
int io_uring_enable_rings(struct io_uring *ring) LIBURING_NOEXCEPT;
int __io_uring_sqring_wait(struct io_uring *ring) LIBURING_NOEXCEPT;
#ifdef _GNU_SOURCE
int io_uring_register_iowq_aff(struct io_uring *ring, size_t cpusz,
const cpu_set_t *mask) LIBURING_NOEXCEPT;
#endif
int io_uring_unregister_iowq_aff(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_register_iowq_max_workers(struct io_uring *ring,
unsigned int *values) LIBURING_NOEXCEPT;
int io_uring_register_ring_fd(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_unregister_ring_fd(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_close_ring_fd(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_register_buf_ring(struct io_uring *ring,
struct io_uring_buf_reg *reg, unsigned int flags) LIBURING_NOEXCEPT;
int io_uring_unregister_buf_ring(struct io_uring *ring, int bgid)
LIBURING_NOEXCEPT;
int io_uring_buf_ring_head(struct io_uring *ring,
int buf_group, uint16_t *head) LIBURING_NOEXCEPT;
int io_uring_register_sync_cancel(struct io_uring *ring,
struct io_uring_sync_cancel_reg *reg)
LIBURING_NOEXCEPT;
int io_uring_register_sync_msg(struct io_uring_sqe *sqe) LIBURING_NOEXCEPT;
int io_uring_register_file_alloc_range(struct io_uring *ring,
unsigned off, unsigned len)
LIBURING_NOEXCEPT;
int io_uring_register_napi(struct io_uring *ring, struct io_uring_napi *napi)
LIBURING_NOEXCEPT;
int io_uring_unregister_napi(struct io_uring *ring, struct io_uring_napi *napi)
LIBURING_NOEXCEPT;
int io_uring_register_ifq(struct io_uring *ring,
struct io_uring_zcrx_ifq_reg *reg) LIBURING_NOEXCEPT;
int io_uring_register_zcrx_ctrl(struct io_uring *ring, struct zcrx_ctrl *ctrl)
LIBURING_NOEXCEPT;
int io_uring_register_clock(struct io_uring *ring,
struct io_uring_clock_register *arg)
LIBURING_NOEXCEPT;
int io_uring_register_bpf_filter(struct io_uring *ring,
struct io_uring_bpf *bpf) LIBURING_NOEXCEPT;
int io_uring_register_bpf_filter_task(struct io_uring_bpf *bpf)
LIBURING_NOEXCEPT;
int io_uring_register_query(struct io_uring_query_hdr *query) LIBURING_NOEXCEPT;
int io_uring_get_events(struct io_uring *ring) LIBURING_NOEXCEPT;
int io_uring_submit_and_get_events(struct io_uring *ring) LIBURING_NOEXCEPT;
/*
* io_uring syscalls.
*/
int io_uring_enter(unsigned int fd, unsigned int to_submit,
unsigned int min_complete, unsigned int flags, sigset_t *sig)
LIBURING_NOEXCEPT;
int io_uring_enter2(unsigned int fd, unsigned int to_submit,
unsigned int min_complete, unsigned int flags,
void *arg, size_t sz) LIBURING_NOEXCEPT;
int io_uring_setup(unsigned int entries, struct io_uring_params *p)
LIBURING_NOEXCEPT;
int io_uring_register(unsigned int fd, unsigned int opcode, const void *arg,
unsigned int nr_args) LIBURING_NOEXCEPT;
/*
* Mapped/registered regions
*/
int io_uring_register_region(struct io_uring *ring,
struct io_uring_mem_region_reg *reg)
LIBURING_NOEXCEPT;
/*
* Mapped buffer ring alloc/register + unregister/free helpers
*/
struct io_uring_buf_ring *io_uring_setup_buf_ring(struct io_uring *ring,
unsigned int nentries,
int bgid, unsigned int flags,
int *err) LIBURING_NOEXCEPT;
int io_uring_free_buf_ring(struct io_uring *ring, struct io_uring_buf_ring *br,
unsigned int nentries, int bgid) LIBURING_NOEXCEPT;
/*
* Helper for the peek/wait single cqe functions. Exported because of that,
* but probably shouldn't be used directly in an application.
*/
int __io_uring_get_cqe(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr, unsigned submit,
unsigned wait_nr, sigset_t *sigmask) LIBURING_NOEXCEPT;
/*
* Enable/disable setting of iowait by the kernel.
*/
int io_uring_set_iowait(struct io_uring *ring, bool enable_iowait)
LIBURING_NOEXCEPT;
#define LIBURING_UDATA_TIMEOUT ((__u64) -1)
/*
* Returns the bit shift needed to index the CQ.
* This shift is 1 for rings with big CQEs, and 0 for rings with normal CQEs.
* CQE `index` can be computed as &cq.cqes[(index & cq.ring_mask) << cqe_shift].
*/
IOURINGINLINE unsigned io_uring_cqe_shift_from_flags(unsigned flags)
LIBURING_NOEXCEPT{
return !!(flags & IORING_SETUP_CQE32);
}
IOURINGINLINE unsigned io_uring_cqe_shift(const struct io_uring *ring)
LIBURING_NOEXCEPT{
return io_uring_cqe_shift_from_flags(ring->flags);
}
IOURINGINLINE unsigned io_uring_cqe_nr(const struct io_uring_cqe *cqe){
const unsigned int shift = !!(cqe->flags & IORING_CQE_F_32);
return 1U << shift;
}
struct io_uring_cqe_iter {
struct io_uring_cqe *cqes;
unsigned mask;
unsigned shift;
unsigned head;
unsigned tail;
};
IOURINGINLINE struct io_uring_cqe_iter
io_uring_cqe_iter_init(const struct io_uring *ring)
LIBURING_NOEXCEPT{
return (struct io_uring_cqe_iter) {
.cqes = ring->cq.cqes,
.mask = ring->cq.ring_mask,
.shift = io_uring_cqe_shift(ring),
.head = *ring->cq.khead,
/* Acquire ordering ensures tail is loaded before any CQEs */
.tail = io_uring_smp_load_acquire(ring->cq.ktail),
};
}
IOURINGINLINE bool io_uring_cqe_iter_next(struct io_uring_cqe_iter *iter,
struct io_uring_cqe **cqe)
LIBURING_NOEXCEPT{
if (iter->head == iter->tail)
return false;
*cqe = &iter->cqes[(iter->head++ & iter->mask) << iter->shift];
if ((*cqe)->flags & IORING_CQE_F_32)
iter->head++;
return true;
}
/*
* NOTE: we should just get rid of the '__head__' being passed in here, it doesn't
* serve a purpose anymore. The below is a bit of a work-around to ensure that
* the compiler doesn't complain about '__head__' being unused (or only written,
* never read), as we use a local iterator for both the head and tail tracking.
*/
#define io_uring_for_each_cqe(ring, __head__, cqe) \
for (struct io_uring_cqe_iter __ITER__ = io_uring_cqe_iter_init(ring); \
(__head__) = __ITER__.head, io_uring_cqe_iter_next(&__ITER__, &(cqe)); \
(void)(__head__))
/*
* Must be called after io_uring_for_each_cqe()
*/
IOURINGINLINE void io_uring_cq_advance(struct io_uring *ring, unsigned nr)
LIBURING_NOEXCEPT{
if (nr) {
struct io_uring_cq *cq = &ring->cq;
/*
* Ensure that the kernel only sees the new value of the head
* index after the CQEs have been read.
*/
io_uring_smp_store_release(cq->khead, *cq->khead + nr);
}
}
/*
* Must be called after io_uring_{peek,wait}_cqe() after the cqe has
* been processed by the application.
*/
IOURINGINLINE void io_uring_cqe_seen(struct io_uring *ring,
struct io_uring_cqe *cqe)
LIBURING_NOEXCEPT{
if (cqe)
io_uring_cq_advance(ring, io_uring_cqe_nr(cqe));
}
/*
* Command prep helpers
*/
/*
* Associate pointer @data with the sqe, for later retrieval from the cqe
* at command completion time with io_uring_cqe_get_data().
*/
IOURINGINLINE void io_uring_sqe_set_data(struct io_uring_sqe *sqe, void *data)
LIBURING_NOEXCEPT{
sqe->user_data = (unsigned long) data;
}
IOURINGINLINE void *io_uring_cqe_get_data(const struct io_uring_cqe *cqe)
LIBURING_NOEXCEPT{
return (void *) (uintptr_t) cqe->user_data;
}
/*
* Assign a 64-bit value to this sqe, which can get retrieved at completion
* time with io_uring_cqe_get_data64. Just like the non-64 variants, except
* these store a 64-bit type rather than a data pointer.
*/
IOURINGINLINE void io_uring_sqe_set_data64(struct io_uring_sqe *sqe,
__u64 data)
LIBURING_NOEXCEPT{
sqe->user_data = data;
}
IOURINGINLINE __u64 io_uring_cqe_get_data64(const struct io_uring_cqe *cqe){
return cqe->user_data;
}
/*
* Tell the app the have the 64-bit variants of the get/set userdata
*/
#define LIBURING_HAVE_DATA64
IOURINGINLINE void io_uring_sqe_set_flags(struct io_uring_sqe *sqe,
unsigned flags)
LIBURING_NOEXCEPT{
sqe->flags = (__u8) flags;
}
IOURINGINLINE void io_uring_sqe_set_buf_group(struct io_uring_sqe *sqe,
int bgid)
LIBURING_NOEXCEPT{
sqe->buf_group = (__u16) bgid;
}
_LOCAL_INLINE void __io_uring_set_target_fixed_file(struct io_uring_sqe *sqe,
unsigned int file_index)
LIBURING_NOEXCEPT{
/* 0 means no fixed files, indexes should be encoded as "index + 1" */
sqe->file_index = file_index + 1;
}
IOURINGINLINE void io_uring_initialize_sqe(struct io_uring_sqe *sqe)
LIBURING_NOEXCEPT{
sqe->flags = 0;
sqe->ioprio = 0;
sqe->rw_flags = 0;
sqe->buf_index = 0;
sqe->personality = 0;
sqe->file_index = 0;
sqe->addr3 = 0;
sqe->__pad2[0] = 0;
}
IOURINGINLINE void io_uring_prep_rw(int op, struct io_uring_sqe *sqe, int fd,
const void *addr, unsigned len,
__u64 offset)
LIBURING_NOEXCEPT{
sqe->opcode = (__u8) op;
sqe->fd = fd;
sqe->off = offset;
sqe->addr = (unsigned long) addr;
sqe->len = len;
}
/*
* io_uring_prep_splice() - Either @fd_in or @fd_out must be a pipe.
*
* - If @fd_in refers to a pipe, @off_in is ignored and must be set to -1.
*
* - If @fd_in does not refer to a pipe and @off_in is -1, then @nbytes are read
* from @fd_in starting from the file offset, which is incremented by the
* number of bytes read.
*
* - If @fd_in does not refer to a pipe and @off_in is not -1, then the starting
* offset of @fd_in will be @off_in.
*
* This splice operation can be used to implement sendfile by splicing to an
* intermediate pipe first, then splice to the final destination.
* In fact, the implementation of sendfile in kernel uses splice internally.
*
* NOTE that even if fd_in or fd_out refers to a pipe, the splice operation
* can still fail with EINVAL if one of the fd doesn't explicitly support splice
* operation, e.g. reading from terminal is unsupported from kernel 5.7 to 5.11.
* Check issue #291 for more information.
*/
IOURINGINLINE void io_uring_prep_splice(struct io_uring_sqe *sqe,
int fd_in, int64_t off_in,
int fd_out, int64_t off_out,
unsigned int nbytes,
unsigned int splice_flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SPLICE, sqe, fd_out, NULL, nbytes,
(__u64) off_out);
sqe->splice_off_in = (__u64) off_in;
sqe->splice_fd_in = fd_in;
sqe->splice_flags = splice_flags;
}
IOURINGINLINE void io_uring_prep_tee(struct io_uring_sqe *sqe,
int fd_in, int fd_out,
unsigned int nbytes,
unsigned int splice_flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_TEE, sqe, fd_out, NULL, nbytes, 0);
sqe->splice_off_in = 0;
sqe->splice_fd_in = fd_in;
sqe->splice_flags = splice_flags;
}
IOURINGINLINE void io_uring_prep_readv(struct io_uring_sqe *sqe, int fd,
const struct iovec *iovecs,
unsigned nr_vecs, __u64 offset)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_READV, sqe, fd, iovecs, nr_vecs, offset);
}
IOURINGINLINE void io_uring_prep_readv2(struct io_uring_sqe *sqe, int fd,
const struct iovec *iovecs,
unsigned nr_vecs, __u64 offset,
int flags)
LIBURING_NOEXCEPT{
io_uring_prep_readv(sqe, fd, iovecs, nr_vecs, offset);
sqe->rw_flags = flags;
}
IOURINGINLINE void io_uring_prep_read_fixed(struct io_uring_sqe *sqe, int fd,
void *buf, unsigned nbytes,
__u64 offset, int buf_index)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_READ_FIXED, sqe, fd, buf, nbytes, offset);
sqe->buf_index = (__u16) buf_index;
}
IOURINGINLINE void io_uring_prep_readv_fixed(struct io_uring_sqe *sqe, int fd,
const struct iovec *iovecs,
unsigned nr_vecs, __u64 offset,
int flags, int buf_index)
LIBURING_NOEXCEPT{
io_uring_prep_readv2(sqe, fd, iovecs, nr_vecs, offset, flags);
sqe->opcode = IORING_OP_READV_FIXED;
sqe->buf_index = (__u16)buf_index;
}
IOURINGINLINE void io_uring_prep_writev(struct io_uring_sqe *sqe, int fd,
const struct iovec *iovecs,
unsigned nr_vecs, __u64 offset)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_WRITEV, sqe, fd, iovecs, nr_vecs, offset);
}
IOURINGINLINE void io_uring_prep_writev2(struct io_uring_sqe *sqe, int fd,
const struct iovec *iovecs,
unsigned nr_vecs, __u64 offset,
int flags)
LIBURING_NOEXCEPT{
io_uring_prep_writev(sqe, fd, iovecs, nr_vecs, offset);
sqe->rw_flags = flags;
}
IOURINGINLINE void io_uring_prep_write_fixed(struct io_uring_sqe *sqe, int fd,
const void *buf, unsigned nbytes,
__u64 offset, int buf_index)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_WRITE_FIXED, sqe, fd, buf, nbytes, offset);
sqe->buf_index = (__u16) buf_index;
}
IOURINGINLINE void io_uring_prep_writev_fixed(struct io_uring_sqe *sqe, int fd,
const struct iovec *iovecs,
unsigned nr_vecs, __u64 offset,
int flags, int buf_index)
LIBURING_NOEXCEPT{
io_uring_prep_writev2(sqe, fd, iovecs, nr_vecs, offset, flags);
sqe->opcode = IORING_OP_WRITEV_FIXED;
sqe->buf_index = (__u16)buf_index;
}
IOURINGINLINE void io_uring_prep_recvmsg(struct io_uring_sqe *sqe, int fd,
struct msghdr *msg, unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_RECVMSG, sqe, fd, msg, 1, 0);
sqe->msg_flags = flags;
}
IOURINGINLINE void io_uring_prep_recvmsg_multishot(struct io_uring_sqe *sqe,
int fd, struct msghdr *msg,
unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_recvmsg(sqe, fd, msg, flags);
sqe->ioprio |= IORING_RECV_MULTISHOT;
}
IOURINGINLINE void io_uring_prep_sendmsg(struct io_uring_sqe *sqe, int fd,
const struct msghdr *msg,
unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SENDMSG, sqe, fd, msg, 1, 0);
sqe->msg_flags = flags;
}
_LOCAL_INLINE unsigned __io_uring_prep_poll_mask(unsigned poll_mask)
LIBURING_NOEXCEPT{
#if __BYTE_ORDER == __BIG_ENDIAN
poll_mask = __swahw32(poll_mask);
#endif
return poll_mask;
}
IOURINGINLINE void io_uring_prep_poll_add(struct io_uring_sqe *sqe, int fd,
unsigned poll_mask)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_POLL_ADD, sqe, fd, NULL, 0, 0);
sqe->poll32_events = __io_uring_prep_poll_mask(poll_mask);
}
IOURINGINLINE void io_uring_prep_poll_multishot(struct io_uring_sqe *sqe,
int fd, unsigned poll_mask)
LIBURING_NOEXCEPT{
io_uring_prep_poll_add(sqe, fd, poll_mask);
sqe->len = IORING_POLL_ADD_MULTI;
}
IOURINGINLINE void io_uring_prep_poll_remove(struct io_uring_sqe *sqe,
__u64 user_data)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_POLL_REMOVE, sqe, -1, NULL, 0, 0);
sqe->addr = user_data;
}
IOURINGINLINE void io_uring_prep_poll_update(struct io_uring_sqe *sqe,
__u64 old_user_data,
__u64 new_user_data,
unsigned poll_mask, unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_POLL_REMOVE, sqe, -1, NULL, flags,
new_user_data);
sqe->addr = old_user_data;
sqe->poll32_events = __io_uring_prep_poll_mask(poll_mask);
}
IOURINGINLINE void io_uring_prep_fsync(struct io_uring_sqe *sqe, int fd,
unsigned fsync_flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FSYNC, sqe, fd, NULL, 0, 0);
sqe->fsync_flags = fsync_flags;
}
IOURINGINLINE void io_uring_prep_nop(struct io_uring_sqe *sqe)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_NOP, sqe, -1, NULL, 0, 0);
}
IOURINGINLINE void io_uring_prep_nop128(struct io_uring_sqe *sqe)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_NOP128, sqe, -1, NULL, 0, 0);
}
IOURINGINLINE void io_uring_prep_timeout(struct io_uring_sqe *sqe,
const struct __kernel_timespec *ts,
unsigned count, unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_TIMEOUT, sqe, -1, ts, 1, count);
sqe->timeout_flags = flags;
}
IOURINGINLINE void io_uring_prep_timeout_remove(struct io_uring_sqe *sqe,
__u64 user_data, unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_TIMEOUT_REMOVE, sqe, -1, NULL, 0, 0);
sqe->addr = user_data;
sqe->timeout_flags = flags;
}
IOURINGINLINE void io_uring_prep_timeout_update(struct io_uring_sqe *sqe,
const struct __kernel_timespec *ts,
__u64 user_data, unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_TIMEOUT_REMOVE, sqe, -1, NULL, 0,
(uintptr_t) ts);
sqe->addr = user_data;
sqe->timeout_flags = flags | IORING_TIMEOUT_UPDATE;
}
IOURINGINLINE void io_uring_prep_accept(struct io_uring_sqe *sqe, int fd,
struct sockaddr *addr,
socklen_t *addrlen, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_ACCEPT, sqe, fd, addr, 0,
uring_ptr_to_u64(addrlen));
sqe->accept_flags = (__u32) flags;
}
/* accept directly into the fixed file table */
IOURINGINLINE void io_uring_prep_accept_direct(struct io_uring_sqe *sqe, int fd,
struct sockaddr *addr,
socklen_t *addrlen, int flags,
unsigned int file_index)
LIBURING_NOEXCEPT{
io_uring_prep_accept(sqe, fd, addr, addrlen, flags);
/* offset by 1 for allocation */
if (file_index == IORING_FILE_INDEX_ALLOC)
file_index--;
__io_uring_set_target_fixed_file(sqe, file_index);
}
IOURINGINLINE void io_uring_prep_multishot_accept(struct io_uring_sqe *sqe,
int fd, struct sockaddr *addr,
socklen_t *addrlen, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_accept(sqe, fd, addr, addrlen, flags);
sqe->ioprio |= IORING_ACCEPT_MULTISHOT;
}
/* multishot accept directly into the fixed file table */
IOURINGINLINE void io_uring_prep_multishot_accept_direct(struct io_uring_sqe *sqe,
int fd,
struct sockaddr *addr,
socklen_t *addrlen,
int flags)
LIBURING_NOEXCEPT{
io_uring_prep_multishot_accept(sqe, fd, addr, addrlen, flags);
__io_uring_set_target_fixed_file(sqe, IORING_FILE_INDEX_ALLOC - 1);
}
IOURINGINLINE void io_uring_prep_cancel64(struct io_uring_sqe *sqe,
__u64 user_data, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_ASYNC_CANCEL, sqe, -1, NULL, 0, 0);
sqe->addr = user_data;
sqe->cancel_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_cancel(struct io_uring_sqe *sqe,
const void *user_data, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_cancel64(sqe, (__u64) (uintptr_t) user_data, flags);
}
IOURINGINLINE void io_uring_prep_cancel_fd(struct io_uring_sqe *sqe, int fd,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_ASYNC_CANCEL, sqe, fd, NULL, 0, 0);
sqe->cancel_flags = (__u32) flags | IORING_ASYNC_CANCEL_FD;
}
IOURINGINLINE void io_uring_prep_link_timeout(struct io_uring_sqe *sqe,
const struct __kernel_timespec *ts,
unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_LINK_TIMEOUT, sqe, -1, ts, 1, 0);
sqe->timeout_flags = flags;
}
IOURINGINLINE void io_uring_prep_connect(struct io_uring_sqe *sqe, int fd,
const struct sockaddr *addr,
socklen_t addrlen)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_CONNECT, sqe, fd, addr, 0, addrlen);
}
IOURINGINLINE void io_uring_prep_bind(struct io_uring_sqe *sqe, int fd,
const struct sockaddr *addr,
socklen_t addrlen)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_BIND, sqe, fd, addr, 0, addrlen);
}
IOURINGINLINE void io_uring_prep_listen(struct io_uring_sqe *sqe, int fd,
int backlog)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_LISTEN, sqe, fd, 0, backlog, 0);
}
struct epoll_event;
IOURINGINLINE void io_uring_prep_epoll_wait(struct io_uring_sqe *sqe, int fd,
struct epoll_event *events,
int maxevents, unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_EPOLL_WAIT, sqe, fd, events, maxevents, 0);
sqe->rw_flags = flags;
}
IOURINGINLINE void io_uring_prep_files_update(struct io_uring_sqe *sqe,
int *fds, unsigned nr_fds,
int offset)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FILES_UPDATE, sqe, -1, fds, nr_fds,
(__u64) offset);
}
IOURINGINLINE void io_uring_prep_fallocate(struct io_uring_sqe *sqe, int fd,
int mode, __u64 offset, __u64 len)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FALLOCATE, sqe, fd,
0, (unsigned int) mode, (__u64) offset);
sqe->addr = (__u64) len;
}
IOURINGINLINE void io_uring_prep_openat(struct io_uring_sqe *sqe, int dfd,
const char *path, int flags,
mode_t mode)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_OPENAT, sqe, dfd, path, mode, 0);
sqe->open_flags = (__u32) flags;
}
/* open directly into the fixed file table */
IOURINGINLINE void io_uring_prep_openat_direct(struct io_uring_sqe *sqe,
int dfd, const char *path,
int flags, mode_t mode,
unsigned file_index)
LIBURING_NOEXCEPT{
io_uring_prep_openat(sqe, dfd, path, flags, mode);
/* offset by 1 for allocation */
if (file_index == IORING_FILE_INDEX_ALLOC)
file_index--;
__io_uring_set_target_fixed_file(sqe, file_index);
}
IOURINGINLINE void io_uring_prep_open(struct io_uring_sqe *sqe,
const char *path, int flags, mode_t mode)
LIBURING_NOEXCEPT{
io_uring_prep_openat(sqe, AT_FDCWD, path, flags, mode);
}
/* open directly into the fixed file table */
IOURINGINLINE void io_uring_prep_open_direct(struct io_uring_sqe *sqe,
const char *path, int flags, mode_t mode,
unsigned file_index)
LIBURING_NOEXCEPT{
io_uring_prep_openat_direct(sqe, AT_FDCWD, path, flags, mode, file_index);
}
IOURINGINLINE void io_uring_prep_close(struct io_uring_sqe *sqe, int fd)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_CLOSE, sqe, fd, NULL, 0, 0);
}
IOURINGINLINE void io_uring_prep_close_direct(struct io_uring_sqe *sqe,
unsigned file_index)
LIBURING_NOEXCEPT{
io_uring_prep_close(sqe, 0);
__io_uring_set_target_fixed_file(sqe, file_index);
}
IOURINGINLINE void io_uring_prep_read(struct io_uring_sqe *sqe, int fd,
void *buf, unsigned nbytes, __u64 offset)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_READ, sqe, fd, buf, nbytes, offset);
}
IOURINGINLINE void io_uring_prep_read_multishot(struct io_uring_sqe *sqe,
int fd, unsigned nbytes,
__u64 offset, int buf_group)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_READ_MULTISHOT, sqe, fd, NULL, nbytes,
offset);
sqe->buf_group = buf_group;
sqe->flags = IOSQE_BUFFER_SELECT;
}
IOURINGINLINE void io_uring_prep_write(struct io_uring_sqe *sqe, int fd,
const void *buf, unsigned nbytes,
__u64 offset)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_WRITE, sqe, fd, buf, nbytes, offset);
}
struct statx;
IOURINGINLINE void io_uring_prep_statx(struct io_uring_sqe *sqe, int dfd,
const char *path, int flags,
unsigned mask, struct statx *statxbuf)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_STATX, sqe, dfd, path, mask,
uring_ptr_to_u64(statxbuf));
sqe->statx_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_fadvise(struct io_uring_sqe *sqe, int fd,
__u64 offset, __u32 len, int advice)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FADVISE, sqe, fd, NULL, (__u32) len, offset);
sqe->fadvise_advice = (__u32) advice;
}
IOURINGINLINE void io_uring_prep_madvise(struct io_uring_sqe *sqe, void *addr,
__u32 length, int advice)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_MADVISE, sqe, -1, addr, (__u32) length, 0);
sqe->fadvise_advice = (__u32) advice;
}
IOURINGINLINE void io_uring_prep_fadvise64(struct io_uring_sqe *sqe, int fd,
__u64 offset, off_t len, int advice)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FADVISE, sqe, fd, NULL, 0, offset);
sqe->addr = len;
sqe->fadvise_advice = (__u32) advice;
}
IOURINGINLINE void io_uring_prep_madvise64(struct io_uring_sqe *sqe, void *addr,
off_t length, int advice)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_MADVISE, sqe, -1, addr, 0, length);
sqe->fadvise_advice = (__u32) advice;
}
IOURINGINLINE void io_uring_prep_send(struct io_uring_sqe *sqe, int sockfd,
const void *buf, size_t len, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SEND, sqe, sockfd, buf, (__u32) len, 0);
sqe->msg_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_send_bundle(struct io_uring_sqe *sqe,
int sockfd, size_t len, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_send(sqe, sockfd, NULL, len, flags);
sqe->ioprio |= IORING_RECVSEND_BUNDLE;
}
IOURINGINLINE void io_uring_prep_send_set_addr(struct io_uring_sqe *sqe,
const struct sockaddr *dest_addr,
__u16 addr_len)
LIBURING_NOEXCEPT{
sqe->addr2 = (unsigned long)(const void *)dest_addr;
sqe->addr_len = addr_len;
}
IOURINGINLINE void io_uring_prep_sendto(struct io_uring_sqe *sqe, int sockfd,
const void *buf, size_t len, int flags,
const struct sockaddr *addr,
socklen_t addrlen)
LIBURING_NOEXCEPT{
io_uring_prep_send(sqe, sockfd, buf, len, flags);
io_uring_prep_send_set_addr(sqe, addr, addrlen);
}
IOURINGINLINE void io_uring_prep_send_zc(struct io_uring_sqe *sqe, int sockfd,
const void *buf, size_t len, int flags,
unsigned zc_flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SEND_ZC, sqe, sockfd, buf, (__u32) len, 0);
sqe->msg_flags = (__u32) flags;
sqe->ioprio = zc_flags;
}
IOURINGINLINE void io_uring_prep_send_zc_fixed(struct io_uring_sqe *sqe,
int sockfd, const void *buf,
size_t len, int flags,
unsigned zc_flags,
unsigned buf_index)
LIBURING_NOEXCEPT{
io_uring_prep_send_zc(sqe, sockfd, buf, len, flags, zc_flags);
sqe->ioprio |= IORING_RECVSEND_FIXED_BUF;
sqe->buf_index = buf_index;
}
IOURINGINLINE void io_uring_prep_sendmsg_zc(struct io_uring_sqe *sqe, int fd,
const struct msghdr *msg,
unsigned flags)
LIBURING_NOEXCEPT{
io_uring_prep_sendmsg(sqe, fd, msg, flags);
sqe->opcode = IORING_OP_SENDMSG_ZC;
}
IOURINGINLINE void io_uring_prep_sendmsg_zc_fixed(struct io_uring_sqe *sqe,
int fd,
const struct msghdr *msg,
unsigned flags,
unsigned buf_index)
LIBURING_NOEXCEPT{
io_uring_prep_sendmsg_zc(sqe, fd, msg, flags);
sqe->ioprio |= IORING_RECVSEND_FIXED_BUF;
sqe->buf_index = buf_index;
}
IOURINGINLINE void io_uring_prep_recv(struct io_uring_sqe *sqe, int sockfd,
void *buf, size_t len, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_RECV, sqe, sockfd, buf, (__u32) len, 0);
sqe->msg_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_recv_multishot(struct io_uring_sqe *sqe,
int sockfd, void *buf,
size_t len, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_recv(sqe, sockfd, buf, len, flags);
sqe->ioprio |= IORING_RECV_MULTISHOT;
}
IOURINGINLINE struct io_uring_recvmsg_out *
io_uring_recvmsg_validate(void *buf, int buf_len, struct msghdr *msgh)
LIBURING_NOEXCEPT{
unsigned long ulen = (unsigned long)(unsigned int)buf_len;
unsigned long hdr = sizeof(struct io_uring_recvmsg_out);
unsigned long namelen = msgh->msg_namelen;
unsigned long controllen = msgh->msg_controllen;
if (buf_len < 0 || ulen < hdr)
return NULL;
/* check each addition separately to avoid integer overflow */
if (namelen > ulen - hdr)
return NULL;
if (controllen > ulen - hdr - namelen)
return NULL;
return (struct io_uring_recvmsg_out *)buf;
}
IOURINGINLINE void *io_uring_recvmsg_name(struct io_uring_recvmsg_out *o)
LIBURING_NOEXCEPT{
return (void *) &o[1];
}
IOURINGINLINE struct cmsghdr *
io_uring_recvmsg_cmsg_firsthdr(struct io_uring_recvmsg_out *o,
struct msghdr *msgh)
LIBURING_NOEXCEPT{
if (o->controllen < sizeof(struct cmsghdr))
return NULL;
return (struct cmsghdr *)((unsigned char *) io_uring_recvmsg_name(o) +
msgh->msg_namelen);
}
IOURINGINLINE struct cmsghdr *
io_uring_recvmsg_cmsg_nexthdr(struct io_uring_recvmsg_out *o, struct msghdr *msgh,
struct cmsghdr *cmsg)
LIBURING_NOEXCEPT{
unsigned char *end;
if (cmsg->cmsg_len < sizeof(struct cmsghdr))
return NULL;
end = (unsigned char *) io_uring_recvmsg_cmsg_firsthdr(o, msgh) +
o->controllen;
cmsg = (struct cmsghdr *)((unsigned char *) cmsg +
CMSG_ALIGN(cmsg->cmsg_len));
if ((unsigned char *) (cmsg + 1) > end)
return NULL;
if (((unsigned char *) cmsg) + CMSG_ALIGN(cmsg->cmsg_len) > end)
return NULL;
return cmsg;
}
IOURINGINLINE void *io_uring_recvmsg_payload(struct io_uring_recvmsg_out *o,
struct msghdr *msgh)
LIBURING_NOEXCEPT{
return (void *)((unsigned char *)io_uring_recvmsg_name(o) +
msgh->msg_namelen + msgh->msg_controllen);
}
IOURINGINLINE unsigned int
io_uring_recvmsg_payload_length(struct io_uring_recvmsg_out *o,
int buf_len, struct msghdr *msgh)
LIBURING_NOEXCEPT{
unsigned long payload_start, payload_end;
if (buf_len < 0)
return 0;
payload_start = (unsigned long) io_uring_recvmsg_payload(o, msgh);
payload_end = (unsigned long) o + buf_len;
if (payload_start >= payload_end)
return 0;
return (unsigned int) (payload_end - payload_start);
}
IOURINGINLINE void io_uring_prep_openat2(struct io_uring_sqe *sqe, int dfd,
const char *path, const struct open_how *how)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_OPENAT2, sqe, dfd, path, sizeof(*how),
(uint64_t) (uintptr_t) how);
}
/* open directly into the fixed file table */
IOURINGINLINE void io_uring_prep_openat2_direct(struct io_uring_sqe *sqe,
int dfd, const char *path,
const struct open_how *how,
unsigned file_index)
LIBURING_NOEXCEPT{
io_uring_prep_openat2(sqe, dfd, path, how);
/* offset by 1 for allocation */
if (file_index == IORING_FILE_INDEX_ALLOC)
file_index--;
__io_uring_set_target_fixed_file(sqe, file_index);
}
struct epoll_event;
IOURINGINLINE void io_uring_prep_epoll_ctl(struct io_uring_sqe *sqe, int epfd,
int fd, int op,
const struct epoll_event *ev)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_EPOLL_CTL, sqe, epfd, ev,
(__u32) op, (__u32) fd);
}
IOURINGINLINE void io_uring_prep_provide_buffers(struct io_uring_sqe *sqe,
void *addr, int len, int nr,
int bgid, int bid)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_PROVIDE_BUFFERS, sqe, nr, addr, (__u32) len,
(__u64) bid);
sqe->buf_group = (__u16) bgid;
}
IOURINGINLINE void io_uring_prep_remove_buffers(struct io_uring_sqe *sqe,
int nr, int bgid)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_REMOVE_BUFFERS, sqe, nr, NULL, 0, 0);
sqe->buf_group = (__u16) bgid;
}
IOURINGINLINE void io_uring_prep_shutdown(struct io_uring_sqe *sqe, int fd,
int how)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SHUTDOWN, sqe, fd, NULL, (__u32) how, 0);
}
IOURINGINLINE void io_uring_prep_unlinkat(struct io_uring_sqe *sqe, int dfd,
const char *path, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_UNLINKAT, sqe, dfd, path, 0, 0);
sqe->unlink_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_unlink(struct io_uring_sqe *sqe,
const char *path, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_unlinkat(sqe, AT_FDCWD, path, flags);
}
IOURINGINLINE void io_uring_prep_renameat(struct io_uring_sqe *sqe, int olddfd,
const char *oldpath, int newdfd,
const char *newpath, unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_RENAMEAT, sqe, olddfd, oldpath,
(__u32) newdfd,
(uint64_t) (uintptr_t) newpath);
sqe->rename_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_rename(struct io_uring_sqe *sqe,
const char *oldpath,
const char *newpath)
LIBURING_NOEXCEPT{
io_uring_prep_renameat(sqe, AT_FDCWD, oldpath, AT_FDCWD, newpath, 0);
}
IOURINGINLINE void io_uring_prep_sync_file_range(struct io_uring_sqe *sqe,
int fd, unsigned len,
__u64 offset, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SYNC_FILE_RANGE, sqe, fd, NULL, len, offset);
sqe->sync_range_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_mkdirat(struct io_uring_sqe *sqe, int dfd,
const char *path, mode_t mode)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_MKDIRAT, sqe, dfd, path, mode, 0);
}
IOURINGINLINE void io_uring_prep_mkdir(struct io_uring_sqe *sqe,
const char *path, mode_t mode)
LIBURING_NOEXCEPT{
io_uring_prep_mkdirat(sqe, AT_FDCWD, path, mode);
}
IOURINGINLINE void io_uring_prep_symlinkat(struct io_uring_sqe *sqe,
const char *target, int newdirfd,
const char *linkpath)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SYMLINKAT, sqe, newdirfd, target, 0,
(uint64_t) (uintptr_t) linkpath);
}
IOURINGINLINE void io_uring_prep_symlink(struct io_uring_sqe *sqe,
const char *target,
const char *linkpath)
LIBURING_NOEXCEPT{
io_uring_prep_symlinkat(sqe, target, AT_FDCWD, linkpath);
}
IOURINGINLINE void io_uring_prep_linkat(struct io_uring_sqe *sqe, int olddfd,
const char *oldpath, int newdfd,
const char *newpath, int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_LINKAT, sqe, olddfd, oldpath, (__u32) newdfd,
(uint64_t) (uintptr_t) newpath);
sqe->hardlink_flags = (__u32) flags;
}
IOURINGINLINE void io_uring_prep_link(struct io_uring_sqe *sqe,
const char *oldpath, const char *newpath,
int flags)
LIBURING_NOEXCEPT{
io_uring_prep_linkat(sqe, AT_FDCWD, oldpath, AT_FDCWD, newpath, flags);
}
IOURINGINLINE void io_uring_prep_msg_ring_cqe_flags(struct io_uring_sqe *sqe,
int fd, unsigned int len, __u64 data,
unsigned int flags, unsigned int cqe_flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_MSG_RING, sqe, fd, NULL, len, data);
sqe->msg_ring_flags = IORING_MSG_RING_FLAGS_PASS | flags;
sqe->file_index = cqe_flags;
}
IOURINGINLINE void io_uring_prep_msg_ring(struct io_uring_sqe *sqe, int fd,
unsigned int len, __u64 data,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_MSG_RING, sqe, fd, NULL, len, data);
sqe->msg_ring_flags = flags;
}
IOURINGINLINE void io_uring_prep_msg_ring_fd(struct io_uring_sqe *sqe, int fd,
int source_fd, int target_fd,
__u64 data, unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_MSG_RING, sqe, fd,
(void *) (uintptr_t) IORING_MSG_SEND_FD, 0, data);
sqe->addr3 = source_fd;
/* offset by 1 for allocation */
if ((unsigned int) target_fd == IORING_FILE_INDEX_ALLOC)
target_fd--;
__io_uring_set_target_fixed_file(sqe, target_fd);
sqe->msg_ring_flags = flags;
}
IOURINGINLINE void io_uring_prep_msg_ring_fd_alloc(struct io_uring_sqe *sqe,
int fd, int source_fd,
__u64 data, unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_msg_ring_fd(sqe, fd, source_fd, IORING_FILE_INDEX_ALLOC,
data, flags);
}
IOURINGINLINE void io_uring_prep_getxattr(struct io_uring_sqe *sqe,
const char *name, char *value,
const char *path, unsigned int len)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_GETXATTR, sqe, 0, name, len,
(__u64) (uintptr_t) value);
sqe->addr3 = (__u64) (uintptr_t) path;
sqe->xattr_flags = 0;
}
IOURINGINLINE void io_uring_prep_setxattr(struct io_uring_sqe *sqe,
const char *name, const char *value,
const char *path, int flags,
unsigned int len)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SETXATTR, sqe, 0, name, len,
(__u64) (uintptr_t) value);
sqe->addr3 = (__u64) (uintptr_t) path;
sqe->xattr_flags = flags;
}
IOURINGINLINE void io_uring_prep_fgetxattr(struct io_uring_sqe *sqe,
int fd, const char *name,
char *value, unsigned int len)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FGETXATTR, sqe, fd, name, len,
(__u64) (uintptr_t) value);
sqe->xattr_flags = 0;
}
IOURINGINLINE void io_uring_prep_fsetxattr(struct io_uring_sqe *sqe, int fd,
const char *name, const char *value,
int flags, unsigned int len)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FSETXATTR, sqe, fd, name, len,
(__u64) (uintptr_t) value);
sqe->xattr_flags = flags;
}
IOURINGINLINE void io_uring_prep_socket(struct io_uring_sqe *sqe, int domain,
int type, int protocol,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SOCKET, sqe, domain, NULL, protocol, type);
sqe->rw_flags = flags;
}
IOURINGINLINE void io_uring_prep_socket_direct(struct io_uring_sqe *sqe,
int domain, int type,
int protocol,
unsigned file_index,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SOCKET, sqe, domain, NULL, protocol, type);
sqe->rw_flags = flags;
/* offset by 1 for allocation */
if (file_index == IORING_FILE_INDEX_ALLOC)
file_index--;
__io_uring_set_target_fixed_file(sqe, file_index);
}
IOURINGINLINE void io_uring_prep_socket_direct_alloc(struct io_uring_sqe *sqe,
int domain, int type,
int protocol,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_SOCKET, sqe, domain, NULL, protocol, type);
sqe->rw_flags = flags;
__io_uring_set_target_fixed_file(sqe, IORING_FILE_INDEX_ALLOC - 1);
}
IOURINGINLINE void __io_uring_prep_uring_cmd(struct io_uring_sqe *sqe,
int op,
__u32 cmd_op,
int fd)
LIBURING_NOEXCEPT{
sqe->opcode = (__u8) op;
sqe->fd = fd;
sqe->cmd_op = cmd_op;
sqe->__pad1 = 0;
sqe->addr = 0ul;
sqe->len = 0;
}
IOURINGINLINE void io_uring_prep_uring_cmd(struct io_uring_sqe *sqe,
int cmd_op,
int fd)
LIBURING_NOEXCEPT{
__io_uring_prep_uring_cmd(sqe, IORING_OP_URING_CMD, cmd_op, fd);
}
IOURINGINLINE void io_uring_prep_uring_cmd128(struct io_uring_sqe *sqe,
int cmd_op,
int fd)
LIBURING_NOEXCEPT{
__io_uring_prep_uring_cmd(sqe, IORING_OP_URING_CMD128, cmd_op, fd);
}
/*
* Prepare commands for sockets
*/
IOURINGINLINE void io_uring_prep_cmd_sock(struct io_uring_sqe *sqe,
int cmd_op,
int fd,
int level,
int optname,
void *optval,
int optlen)
LIBURING_NOEXCEPT{
io_uring_prep_uring_cmd(sqe, cmd_op, fd);
sqe->optval = (unsigned long) (uintptr_t) optval;
sqe->optname = optname;
sqe->optlen = optlen;
sqe->level = level;
}
IOURINGINLINE void io_uring_prep_cmd_getsockname(struct io_uring_sqe *sqe,
int fd, struct sockaddr *sockaddr,
socklen_t *sockaddr_len,
int peer)
LIBURING_NOEXCEPT{
io_uring_prep_uring_cmd(sqe, SOCKET_URING_OP_GETSOCKNAME, fd);
sqe->addr = (uintptr_t) sockaddr;
sqe->addr3 = (unsigned long) (uintptr_t) sockaddr_len;
sqe->optlen = peer;
}
IOURINGINLINE void io_uring_prep_waitid(struct io_uring_sqe *sqe,
idtype_t idtype,
id_t id,
siginfo_t *infop,
int options, unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_WAITID, sqe, id, NULL, (unsigned) idtype, 0);
sqe->waitid_flags = flags;
sqe->file_index = options;
sqe->addr2 = (unsigned long) infop;
}
IOURINGINLINE void io_uring_prep_futex_wake(struct io_uring_sqe *sqe,
const uint32_t *futex, uint64_t val,
uint64_t mask, uint32_t futex_flags,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FUTEX_WAKE, sqe, futex_flags, futex, 0, val);
sqe->futex_flags = flags;
sqe->addr3 = mask;
}
IOURINGINLINE void io_uring_prep_futex_wait(struct io_uring_sqe *sqe,
const uint32_t *futex, uint64_t val,
uint64_t mask, uint32_t futex_flags,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FUTEX_WAIT, sqe, futex_flags, futex, 0, val);
sqe->futex_flags = flags;
sqe->addr3 = mask;
}
struct futex_waitv;
IOURINGINLINE void io_uring_prep_futex_waitv(struct io_uring_sqe *sqe,
const struct futex_waitv *futex,
uint32_t nr_futex,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FUTEX_WAITV, sqe, 0, futex, nr_futex, 0);
sqe->futex_flags = flags;
}
IOURINGINLINE void io_uring_prep_fixed_fd_install(struct io_uring_sqe *sqe,
int fd,
unsigned int flags)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FIXED_FD_INSTALL, sqe, fd, NULL, 0, 0);
sqe->flags = IOSQE_FIXED_FILE;
sqe->install_fd_flags = flags;
}
#ifdef _GNU_SOURCE
IOURINGINLINE void io_uring_prep_ftruncate(struct io_uring_sqe *sqe,
int fd, loff_t len)
LIBURING_NOEXCEPT{
io_uring_prep_rw(IORING_OP_FTRUNCATE, sqe, fd, 0, 0, len);
}
#endif
IOURINGINLINE void io_uring_prep_cmd_discard(struct io_uring_sqe *sqe,
int fd,
uint64_t offset, uint64_t nbytes)
LIBURING_NOEXCEPT{
io_uring_prep_uring_cmd(sqe, BLOCK_URING_CMD_DISCARD, fd);
sqe->addr = offset;
sqe->addr3 = nbytes;
}
IOURINGINLINE void io_uring_prep_pipe(struct io_uring_sqe *sqe, int *fds,
int pipe_flags){
io_uring_prep_rw(IORING_OP_PIPE, sqe, 0, fds, 0, 0);
sqe->pipe_flags = (__u32) pipe_flags;
}
/* setup pipe directly into the fixed file table */
IOURINGINLINE void io_uring_prep_pipe_direct(struct io_uring_sqe *sqe, int *fds,
int pipe_flags,
unsigned int file_index){
io_uring_prep_pipe(sqe, fds, pipe_flags);
/* offset by 1 for allocation */
if (file_index == IORING_FILE_INDEX_ALLOC)
file_index--;
__io_uring_set_target_fixed_file(sqe, file_index);
}
/* Read the kernel's SQ head index with appropriate memory ordering */
IOURINGINLINE unsigned io_uring_load_sq_head(const struct io_uring *ring)
LIBURING_NOEXCEPT{
/*
* Without acquire ordering, we could overwrite a SQE before the kernel
* finished reading it. We don't need the acquire ordering for
* non-SQPOLL since then we drive updates.
*/
if (ring->flags & IORING_SETUP_SQPOLL)
return io_uring_smp_load_acquire(ring->sq.khead);
return *ring->sq.khead;
}
/*
* Returns number of unconsumed (if SQPOLL) or unsubmitted entries exist in
* the SQ ring
*/
IOURINGINLINE unsigned io_uring_sq_ready(const struct io_uring *ring)
LIBURING_NOEXCEPT{
/* always use real head, to avoid losing sync for short submit */
return ring->sq.sqe_tail - io_uring_load_sq_head(ring);
}
/*
* Returns how much space is left in the SQ ring.
*/
IOURINGINLINE unsigned io_uring_sq_space_left(const struct io_uring *ring)
LIBURING_NOEXCEPT{
return ring->sq.ring_entries - io_uring_sq_ready(ring);
}
/*
* Returns the bit shift needed to index the SQ.
* This shift is 1 for rings with big SQEs, and 0 for rings with normal SQEs.
* SQE `index` can be computed as &sq.sqes[(index & sq.ring_mask) << sqe_shift].
*/
IOURINGINLINE unsigned io_uring_sqe_shift_from_flags(unsigned flags)
LIBURING_NOEXCEPT{
return !!(flags & IORING_SETUP_SQE128);
}
IOURINGINLINE unsigned io_uring_sqe_shift(const struct io_uring *ring)
LIBURING_NOEXCEPT{
return io_uring_sqe_shift_from_flags(ring->flags);
}
/*
* Only applicable when using SQPOLL - allows the caller to wait for space
* to free up in the SQ ring, which happens when the kernel side thread has
* consumed one or more entries. If the SQ ring is currently non-full, no
* action is taken. Note: may return -EINVAL if the kernel doesn't support
* this feature.
*/
IOURINGINLINE int io_uring_sqring_wait(struct io_uring *ring)
LIBURING_NOEXCEPT{
if (!(ring->flags & IORING_SETUP_SQPOLL))
return 0;
if (io_uring_sq_space_left(ring))
return 0;
return __io_uring_sqring_wait(ring);
}
/*
* Returns how many unconsumed entries are ready in the CQ ring
*/
IOURINGINLINE unsigned io_uring_cq_ready(const struct io_uring *ring)
LIBURING_NOEXCEPT{
return io_uring_smp_load_acquire(ring->cq.ktail) - *ring->cq.khead;
}
/*
* Returns true if there are overflow entries waiting to be flushed onto
* the CQ ring
*/
IOURINGINLINE bool io_uring_cq_has_overflow(const struct io_uring *ring)
LIBURING_NOEXCEPT{
return IO_URING_READ_ONCE(*ring->sq.kflags) & IORING_SQ_CQ_OVERFLOW;
}
/*
* Returns true if the eventfd notification is currently enabled
*/
IOURINGINLINE bool io_uring_cq_eventfd_enabled(const struct io_uring *ring)
LIBURING_NOEXCEPT{
if (!ring->cq.kflags)
return true;
return !(*ring->cq.kflags & IORING_CQ_EVENTFD_DISABLED);
}
/*
* Toggle eventfd notification on or off, if an eventfd is registered with
* the ring.
*/
IOURINGINLINE int io_uring_cq_eventfd_toggle(struct io_uring *ring,
bool enabled)
LIBURING_NOEXCEPT{
uint32_t flags;
if (enabled == io_uring_cq_eventfd_enabled(ring))
return 0;
if (!ring->cq.kflags)
return -EOPNOTSUPP;
flags = *ring->cq.kflags;
if (enabled)
flags &= ~IORING_CQ_EVENTFD_DISABLED;
else
flags |= IORING_CQ_EVENTFD_DISABLED;
IO_URING_WRITE_ONCE(*ring->cq.kflags, flags);
return 0;
}
/*
* Return an IO completion, waiting for 'wait_nr' completions if one isn't
* readily available. Returns 0 with cqe_ptr filled in on success, -errno on
* failure.
*/
IOURINGINLINE int io_uring_wait_cqe_nr(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr,
unsigned wait_nr)
LIBURING_NOEXCEPT{
return __io_uring_get_cqe(ring, cqe_ptr, 0, wait_nr, NULL);
}
_LOCAL_INLINE bool io_uring_skip_cqe(struct io_uring *ring,
struct io_uring_cqe *cqe, int *err){
if (cqe->flags & IORING_CQE_F_SKIP)
goto out;
if (ring->features & IORING_FEAT_EXT_ARG)
return false;
if (cqe->user_data != LIBURING_UDATA_TIMEOUT)
return false;
if (cqe->res < 0)
*err = cqe->res;
out:
io_uring_cq_advance(ring, io_uring_cqe_nr(cqe));
return !*err;
}
/*
* Internal helper, don't use directly in applications. Use one of the
* "official" versions of this, io_uring_peek_cqe(), io_uring_wait_cqe(),
* or io_uring_wait_cqes*().
*/
IOURINGINLINE int __io_uring_peek_cqe(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr,
unsigned *nr_available)
LIBURING_NOEXCEPT{
struct io_uring_cqe *cqe;
int err = 0;
unsigned available;
unsigned mask = ring->cq.ring_mask;
unsigned shift = io_uring_cqe_shift(ring);
do {
unsigned tail = io_uring_smp_load_acquire(ring->cq.ktail);
/*
* The acquire ordering on the tail load pairs with the kernel
* side publishing CQEs, and guarantees the contents of any
* entry in [head, tail). The CQ head is only ever written by
* the application, so a plain load is sufficient.
*/
unsigned head = *ring->cq.khead;
cqe = NULL;
available = tail - head;
if (!available)
break;
cqe = &ring->cq.cqes[(head & mask) << shift];
if (!io_uring_skip_cqe(ring, cqe, &err)) {
/*
* If an error was set, the CQE was an internal
* timeout and has already been consumed - don't
* return a pointer to it.
*/
if (err)
cqe = NULL;
break;
}
cqe = NULL;
} while (1);
*cqe_ptr = cqe;
if (nr_available)
*nr_available = available;
return err;
}
/*
* Return an IO completion, if one is readily available. Returns 0 with
* cqe_ptr filled in on success, -errno on failure.
*/
IOURINGINLINE int io_uring_peek_cqe(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr)
LIBURING_NOEXCEPT{
if (!__io_uring_peek_cqe(ring, cqe_ptr, NULL)) {
if (*cqe_ptr)
return 0;
/*
* If the CQ is empty and there's nothing the kernel could
* flush to it (no IOPOLL completions to reap, no overflown
* CQEs, no pending task work), avoid the round trip into
* the full get_cqe machinery.
*/
if (!(ring->flags & IORING_SETUP_IOPOLL) &&
!(IO_URING_READ_ONCE(*ring->sq.kflags) &
(IORING_SQ_CQ_OVERFLOW | IORING_SQ_TASKRUN)))
return -EAGAIN;
}
return io_uring_wait_cqe_nr(ring, cqe_ptr, 0);
}
/*
* Return an IO completion, waiting for it if necessary. Returns 0 with
* cqe_ptr filled in on success, -errno on failure.
*/
IOURINGINLINE int io_uring_wait_cqe(struct io_uring *ring,
struct io_uring_cqe **cqe_ptr)
LIBURING_NOEXCEPT{
if (!__io_uring_peek_cqe(ring, cqe_ptr, NULL) && *cqe_ptr)
return 0;
return io_uring_wait_cqe_nr(ring, cqe_ptr, 1);
}
/*
* Return an sqe to fill. Application must later call io_uring_submit()
* when it's ready to tell the kernel about it. The caller may call this
* function multiple times before calling io_uring_submit().
*
* Returns a vacant sqe, or NULL if we're full.
*/
IOURINGINLINE struct io_uring_sqe *_io_uring_get_sqe(struct io_uring *ring)
LIBURING_NOEXCEPT{
struct io_uring_sq *sq = &ring->sq;
unsigned head = io_uring_load_sq_head(ring), tail = sq->sqe_tail;
struct io_uring_sqe *sqe;
if (tail - head >= sq->ring_entries)
return NULL;
sqe = &sq->sqes[(tail & sq->ring_mask) << io_uring_sqe_shift(ring)];
sq->sqe_tail = tail + 1;
io_uring_initialize_sqe(sqe);
return sqe;
}
/*
* Return the appropriate mask for a buffer ring of size 'ring_entries'
*/
IOURINGINLINE int io_uring_buf_ring_mask(__u32 ring_entries)
LIBURING_NOEXCEPT{
return ring_entries - 1;
}
IOURINGINLINE void io_uring_buf_ring_init(struct io_uring_buf_ring *br)
LIBURING_NOEXCEPT{
br->tail = 0;
}
/*
* Assign 'buf' with the addr/len/buffer ID supplied
*/
IOURINGINLINE void io_uring_buf_ring_add(struct io_uring_buf_ring *br,
void *addr, unsigned int len,
unsigned short bid, int mask,
int buf_offset)
LIBURING_NOEXCEPT{
struct io_uring_buf *buf = &br->bufs[(br->tail + buf_offset) & mask];
buf->addr = (unsigned long) (uintptr_t) addr;
buf->len = len;
buf->bid = bid;
}
/*
* Make 'count' new buffers visible to the kernel. Called after
* io_uring_buf_ring_add() has been called 'count' times to fill in new
* buffers.
*/
IOURINGINLINE void io_uring_buf_ring_advance(struct io_uring_buf_ring *br,
int count)
LIBURING_NOEXCEPT{
unsigned short new_tail = br->tail + count;
io_uring_smp_store_release(&br->tail, new_tail);
}
IOURINGINLINE void __io_uring_buf_ring_cq_advance(struct io_uring *ring,
struct io_uring_buf_ring *br,
int cq_count, int buf_count)
LIBURING_NOEXCEPT{
io_uring_buf_ring_advance(br, buf_count);
io_uring_cq_advance(ring, cq_count);
}
/*
* Make 'count' new buffers visible to the kernel while at the same time
* advancing the CQ ring seen entries. This can be used when the application
* is using ring provided buffers and returns buffers while processing CQEs,
* avoiding an extra atomic when needing to increment both the CQ ring and
* the ring buffer index at the same time.
*/
IOURINGINLINE void io_uring_buf_ring_cq_advance(struct io_uring *ring,
struct io_uring_buf_ring *br,
int count)
LIBURING_NOEXCEPT{
__io_uring_buf_ring_cq_advance(ring, br, count, count);
}
IOURINGINLINE int io_uring_buf_ring_available(struct io_uring *ring,
struct io_uring_buf_ring *br,
unsigned short bgid)
LIBURING_NOEXCEPT{
uint16_t head;
int ret;
ret = io_uring_buf_ring_head(ring, bgid, &head);
if (ret)
return ret;
return (uint16_t) (br->tail - head);
}
/*
* As of liburing-2.2, io_uring_get_sqe() has been converted into a
* "static inline" function. However, this change breaks seamless
* updates of liburing.so, as applications would need to be recompiled.
* To ensure backward compatibility, liburing keeps the original
* io_uring_get_sqe() symbol available in the shared library.
*
* To accomplish this, io_uring_get_sqe() is defined as a non-static
* inline function when LIBURING_INTERNAL is set, which only applies
* during liburing.so builds.
*
* This strategy ensures new users adopt the "static inline" version
* while preserving compatibility for old applications linked against
* the shared library.
*
* Relevant commits:
* 8be8af4afcb4 ("queue: provide io_uring_get_sqe() symbol again")
* 52dcdbba35c8 ("src/queue: protect io_uring_get_sqe() with LIBURING_INTERNAL")
*/
#ifndef LIBURING_INTERNAL
IOURINGINLINE struct io_uring_sqe *io_uring_get_sqe(struct io_uring *ring)
LIBURING_NOEXCEPT{
return _io_uring_get_sqe(ring);
}
#else
struct io_uring_sqe *io_uring_get_sqe(struct io_uring *ring);
#endif
/*
* Return a 128B sqe to fill. Applications must later call io_uring_submit()
* when it's ready to tell the kernel about it. The caller may call this
* function multiple times before calling io_uring_submit().
*
* Returns a vacant 128B sqe, or NULL if we're full. If the current tail is the
* last entry in the ring, this function will insert a nop + skip complete such
* that the 128b entry wraps back to the beginning of the queue for a
* contiguous big sq entry. It's up to the caller to use a 128b opcode in order
* for the kernel to know how to advance its sq head pointer.
*/
IOURINGINLINE struct io_uring_sqe *io_uring_get_sqe128(struct io_uring *ring)
LIBURING_NOEXCEPT{
struct io_uring_sq *sq = &ring->sq;
unsigned head = io_uring_load_sq_head(ring), tail = sq->sqe_tail;
struct io_uring_sqe *sqe;
if (ring->flags & IORING_SETUP_SQE128)
return io_uring_get_sqe(ring);
if (!(ring->flags & IORING_SETUP_SQE_MIXED))
return NULL;
if (((tail + 1) & sq->ring_mask) == 0) {
if ((tail + 2) - head >= sq->ring_entries)
return NULL;
sqe = _io_uring_get_sqe(ring);
io_uring_prep_nop(sqe);
sqe->flags |= IOSQE_CQE_SKIP_SUCCESS;
tail = sq->sqe_tail;
} else if ((tail + 1) - head >= sq->ring_entries) {
return NULL;
}
sqe = &sq->sqes[tail & sq->ring_mask];
sq->sqe_tail = tail + 2;
io_uring_initialize_sqe(sqe);
return sqe;
}
ssize_t io_uring_mlock_size(unsigned entries, unsigned flags)
LIBURING_NOEXCEPT;
ssize_t io_uring_mlock_size_params(unsigned entries, struct io_uring_params *p)
LIBURING_NOEXCEPT;
ssize_t io_uring_memory_size(unsigned entries, unsigned flags)
LIBURING_NOEXCEPT;
ssize_t io_uring_memory_size_params(unsigned entries, struct io_uring_params *p)
LIBURING_NOEXCEPT;
/*
* Versioning information for liburing.
*
* Use IO_URING_CHECK_VERSION() for compile time checks including from
* preprocessor directives.
*
* Use io_uring_check_version() for runtime checks of the version of
* liburing that was loaded by the dynamic linker.
*/
int io_uring_major_version(void) LIBURING_NOEXCEPT;
int io_uring_minor_version(void) LIBURING_NOEXCEPT;
bool io_uring_check_version(int major, int minor) LIBURING_NOEXCEPT;
#define IO_URING_CHECK_VERSION(major,minor) \
(major > IO_URING_VERSION_MAJOR || \
(major == IO_URING_VERSION_MAJOR && \
minor > IO_URING_VERSION_MINOR))
#ifdef __cplusplus
}
#endif
#ifdef IOURINGINLINE
#undef IOURINGINLINE
#endif
#ifdef _LOCAL_INLINE
#undef _LOCAL_INLINE
#endif
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_SYSCALL_H
#define LIBURING_SYSCALL_H
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <liburing.h>
/*
* Don't put this below the #include "arch/$arch/syscall.h", that
* file may need it.
*/
struct io_uring_params;
static inline void *ERR_PTR(intptr_t n){
return (void *) n;
}
static inline int PTR_ERR(const void *ptr){
return (int) (intptr_t) ptr;
}
static inline bool IS_ERR(const void *ptr){
return uring_unlikely((uintptr_t) ptr >= (uintptr_t) -4095UL);
}
#if defined(__x86_64__) || defined(__i386__)
#include "arch/x86/syscall.h"
#elif defined(__aarch64__)
#include "arch/aarch64/syscall.h"
#elif defined(__riscv) && __riscv_xlen == 64
#include "arch/riscv64/syscall.h"
#else
/*
* We don't have native syscall wrappers
* for this arch. Must use libc!
*/
#ifdef CONFIG_NOLIBC
#error "This arch doesn't support building liburing without libc"
#endif
/* libc syscall wrappers. */
#include "arch/generic/syscall.h"
#endif
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_INT_FLAGS
#define LIBURING_INT_FLAGS
#define INT_FLAGS_MASK (IORING_ENTER_REGISTERED_RING | \
IORING_ENTER_NO_IOWAIT)
enum {
INT_FLAG_REG_RING = IORING_ENTER_REGISTERED_RING,
INT_FLAG_NO_IOWAIT = IORING_ENTER_NO_IOWAIT,
INT_FLAG_REG_REG_RING = 1,
INT_FLAG_APP_MEM = 2,
INT_FLAG_CQ_ENTER = 4,
};
static inline int ring_enter_flags(struct io_uring *ring){
return ring->int_flags & INT_FLAGS_MASK;
}
#endif
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_LIB_H
#define LIBURING_LIB_H
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include "arch/x86/lib.h"
#elif defined(__aarch64__)
#include "arch/aarch64/lib.h"
#elif defined(__riscv) && __riscv_xlen == 64
#include "arch/riscv64/lib.h"
#else
/*
* We don't have nolibc support for this arch. Must use libc!
*/
#ifdef CONFIG_NOLIBC
#error "This arch doesn't support building liburing without libc"
#endif
/* libc wrappers. */
#include "arch/generic/lib.h"
#endif
#ifndef offsetof
#define offsetof(TYPE, FIELD) ((size_t) &((TYPE *)0)->FIELD)
#endif
#ifndef container_of
#define container_of(PTR, TYPE, MEMBER) \
((TYPE *)((char *)(PTR) - __builtin_offsetof(TYPE, MEMBER)))
#endif
#define __maybe_unused __attribute__((__unused__))
#define __hot __attribute__((__hot__))
#define __cold __attribute__((__cold__))
#ifdef CONFIG_NOLIBC
void *__uring_memset(void *s, int c, size_t n);
void *__uring_malloc(size_t len);
void __uring_free(void *p);
#define malloc(LEN) __uring_malloc(LEN)
#define free(PTR) __uring_free(PTR)
#define memset(PTR, C, LEN) __uring_memset(PTR, C, LEN)
#endif
#endif
/* #ifndef LIBURING_LIB_H */
/* SPDX-License-Identifier: MIT */
#ifndef LIBURING_SETUP_H
#define LIBURING_SETUP_H
int __io_uring_queue_init_params(unsigned entries, struct io_uring *ring,
struct io_uring_params *p, void *buf,
size_t buf_size);
void io_uring_unmap_rings(struct io_uring_sq *sq, struct io_uring_cq *cq);
int io_uring_mmap(int fd, struct io_uring_params *p, struct io_uring_sq *sq,
struct io_uring_cq *cq);
void io_uring_setup_ring_pointers(struct io_uring_params *p,
struct io_uring_sq *sq,
struct io_uring_cq *cq);
#endif
/* SPDX-License-Identifier: MIT */
/*
* Description: Helpers for tests.
*/
#ifndef LIBURING_HELPERS_H
#define LIBURING_HELPERS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "liburing.h"
#include "../src/setup.h"
#include <arpa/inet.h>
#include <sys/time.h>
#include <stdlib.h>
enum t_setup_ret {
T_SETUP_OK = 0,
T_SETUP_SKIP,
};
enum t_test_result {
T_EXIT_PASS = 0,
T_EXIT_FAIL = 1,
T_EXIT_SKIP = 77,
};
/*
* Some Android versions lack aligned_alloc in stdlib.h.
* To avoid making large changes in tests, define a helper
* function that wraps posix_memalign as our own aligned_alloc.
*/
void *t_aligned_alloc(size_t alignment, size_t size);
/*
* Helper for binding socket to an ephemeral port.
* The port number to be bound is returned in @addr->sin_port.
*/
int t_bind_ephemeral_port(int fd, struct sockaddr_in *addr);
/*
* Helper for allocating memory in tests.
*/
void *t_malloc(size_t size);
/*
* Helper for allocating size bytes aligned on a boundary.
*/
void t_posix_memalign(void **memptr, size_t alignment, size_t size);
/*
* Helper for allocating space for an array of nmemb elements
* with size bytes for each element.
*/
void *t_calloc(size_t nmemb, size_t size);
/*
* Helper for creating file and write @size byte buf with 0xaa value in the file.
*/
void t_create_file(const char *file, size_t size);
/*
* Helper for creating file and write @size byte buf with @pattern value in
* the file.
*/
void t_create_file_pattern(const char *file, size_t size, char pattern);
/*
* Helper for creating @buf_num number of iovec
* with @buf_size bytes buffer of each iovec.
*/
struct iovec *t_create_buffers(size_t buf_num, size_t buf_size);
/*
* Helper for creating connected socket pairs
*/
int t_create_socket_pair(int fd[2], bool stream);
int t_create_socketpair_ip(struct sockaddr_storage *addr,
int *sock_client, int *sock_server,
bool ipv6, bool client_connect,
bool msg_zc, bool tcp, const char *name);
/*
* Helper for setting up a ring and checking for user privs
*/
enum t_setup_ret t_create_ring_params(int depth, struct io_uring *ring,
struct io_uring_params *p);
enum t_setup_ret t_create_ring(int depth, struct io_uring *ring,
unsigned int flags);
enum t_setup_ret t_register_buffers(struct io_uring *ring,
const struct iovec *iovecs,
unsigned nr_iovecs);
bool t_probe_defer_taskrun(void);
void t_set_nonblock(int fd);
void t_clear_nonblock(int fd);
unsigned __io_uring_flush_sq(struct io_uring *ring);
static inline int t_io_uring_init_sqarray(unsigned entries, struct io_uring *ring,
struct io_uring_params *p){
int ret;
ret = __io_uring_queue_init_params(entries, ring, p, NULL, 0);
return ret >= 0 ? 0 : ret;
}
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
void t_error(int status, int errnum, const char *format, ...);
unsigned long long mtime_since(const struct timeval *s, const struct timeval *e);
unsigned long long mtime_since_now(struct timeval *tv);
unsigned long long utime_since(const struct timeval *s, const struct timeval *e);
unsigned long long utime_since_now(struct timeval *tv);
int t_submit_and_wait_single(struct io_uring *ring, struct io_uring_cqe **cqe);
size_t t_iovec_data_length(struct iovec *iov, unsigned iov_len);
unsigned long t_compare_data_iovec(struct iovec *iov_src, unsigned nr_src,
struct iovec *iov_dst, unsigned nr_dst);
#ifdef __cplusplus
}
#endif
#endif
/* SPDX-License-Identifier: MIT */
/*
* Description: Helpers for NVMe uring passthrough commands
*/
#ifndef LIBURING_NVME_H
#define LIBURING_NVME_H
#ifdef __cplusplus
extern "C" {
#endif
#include <sys/ioctl.h>
#include <linux/nvme_ioctl.h>
/*
* If the uapi headers installed on the system lacks nvme uring command
* support, use the local version to prevent compilation issues.
*/
#ifndef CONFIG_HAVE_NVME_URING
struct nvme_uring_cmd {
__u8 opcode;
__u8 flags;
__u16 rsvd1;
__u32 nsid;
__u32 cdw2;
__u32 cdw3;
__u64 metadata;
__u64 addr;
__u32 metadata_len;
__u32 data_len;
__u32 cdw10;
__u32 cdw11;
__u32 cdw12;
__u32 cdw13;
__u32 cdw14;
__u32 cdw15;
__u32 timeout_ms;
__u32 rsvd2;
};
#define NVME_URING_CMD_IO _IOWR('N', 0x80, struct nvme_uring_cmd)
#define NVME_URING_CMD_IO_VEC _IOWR('N', 0x81, struct nvme_uring_cmd)
#endif
/* CONFIG_HAVE_NVME_URING */
#define NVME_DEFAULT_IOCTL_TIMEOUT 0
#define NVME_IDENTIFY_DATA_SIZE 4096
#define NVME_IDENTIFY_CSI_SHIFT 24
#define NVME_IDENTIFY_CNS_NS 0
#define NVME_CSI_NVM 0
enum nvme_admin_opcode {
nvme_admin_identify = 0x06,
};
enum nvme_io_opcode {
nvme_cmd_write = 0x01,
nvme_cmd_read = 0x02,
};
static int nsid;
static __u32 lba_shift;
static __u32 meta_size;
struct nvme_lbaf {
__le16 ms;
__u8 ds;
__u8 rp;
};
struct nvme_id_ns {
__le64 nsze;
__le64 ncap;
__le64 nuse;
__u8 nsfeat;
__u8 nlbaf;
__u8 flbas;
__u8 mc;
__u8 dpc;
__u8 dps;
__u8 nmic;
__u8 rescap;
__u8 fpi;
__u8 dlfeat;
__le16 nawun;
__le16 nawupf;
__le16 nacwu;
__le16 nabsn;
__le16 nabo;
__le16 nabspf;
__le16 noiob;
__u8 nvmcap[16];
__le16 npwg;
__le16 npwa;
__le16 npdg;
__le16 npda;
__le16 nows;
__le16 mssrl;
__le32 mcl;
__u8 msrc;
__u8 rsvd81[11];
__le32 anagrpid;
__u8 rsvd96[3];
__u8 nsattr;
__le16 nvmsetid;
__le16 endgid;
__u8 nguid[16];
__u8 eui64[8];
struct nvme_lbaf lbaf[16];
__u8 rsvd192[192];
__u8 vs[3712];
};
static inline int ilog2(uint32_t i){
int log = -1;
while (i) {
i >>= 1;
log++;
}
return log;
}
__attribute__((__unused__))
static int nvme_get_info(const char *file){
struct nvme_id_ns ns;
int fd, err;
__u32 lba_size;
fd = open(file, O_RDONLY);
if (fd < 0) {
perror("file open");
return -errno;
}
nsid = ioctl(fd, NVME_IOCTL_ID);
if (nsid < 0) {
close(fd);
return -errno;
}
struct nvme_passthru_cmd cmd = {
.opcode = nvme_admin_identify,
.nsid = nsid,
.addr = (__u64)(uintptr_t)&ns,
.data_len = NVME_IDENTIFY_DATA_SIZE,
.cdw10 = NVME_IDENTIFY_CNS_NS,
.cdw11 = NVME_CSI_NVM << NVME_IDENTIFY_CSI_SHIFT,
.timeout_ms = NVME_DEFAULT_IOCTL_TIMEOUT,
};
err = ioctl(fd, NVME_IOCTL_ADMIN_CMD, &cmd);
if (err) {
close(fd);
return err;
}
lba_size = 1 << ns.lbaf[(ns.flbas & 0x0f)].ds;
lba_shift = ilog2(lba_size);
meta_size = ns.lbaf[(ns.flbas & 0x0f)].ms;
close(fd);
return 0;
}
#ifdef __cplusplus
}
#endif
#endif
/* SPDX-License-Identifier: MIT */
/*
* Description: Test configs for tests.
*/
#ifndef LIBURING_TEST_H
#define LIBURING_TEST_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct io_uring_test_config {
unsigned int flags;
const char *description;
} io_uring_test_config;
__attribute__((__unused__))
static io_uring_test_config io_uring_test_configs[] = {
{ 0, "default" },
{ IORING_SETUP_SQE128, "large SQE"},
{ IORING_SETUP_CQE32, "large CQE"},
{ IORING_SETUP_SQE128 | IORING_SETUP_CQE32, "large SQE/CQE" },
{ IORING_SETUP_SQ_REWIND, "rewind SQ"},
{ IORING_SETUP_SQ_REWIND | IORING_SETUP_SQE128, "large rewind SQ"},
};
#define FOR_ALL_TEST_CONFIGS \
for (int i = 0; i < sizeof(io_uring_test_configs) / sizeof(io_uring_test_configs[0]); i++)
#define IORING_GET_TEST_CONFIG_FLAGS() (io_uring_test_configs[i].flags)
#define IORING_GET_TEST_CONFIG_DESCRIPTION() (io_uring_test_configs[i].description)
#ifdef __cplusplus
}
#endif
#endif
#ifndef T_LIBURING_BPF_DEFS_H_
#define T_LIBURING_BPF_DEFS_H_
#include <linux/types.h>
#include <linux/errno.h>
#include <stddef.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "liburing/io_uring.h"
struct io_ring_ctx {};
struct iou_loop_params {
__u32 cq_wait_idx;
};
enum {
IOU_REGION_MEM = 0,
IOU_REGION_CQ = 1,
IOU_REGION_SQ = 2,
};
enum {
IOU_LOOP_CONTINUE = 0,
IOU_LOOP_STOP = 1,
};
struct io_uring_bpf_ops {
int (*loop_step)(struct io_ring_ctx *, struct iou_loop_params *);
__u32 ring_fd;
};
extern __u8 *bpf_io_uring_get_region(struct io_ring_ctx *ctx, __u32 region_id, const size_t rdwr_buf_size) __weak __ksym;
extern int bpf_io_uring_submit_sqes(struct io_ring_ctx *ctx, __u32 nr) __weak __ksym;
#endif
/* T_LIBURING_BPF_DEFS_H_ */
#ifndef LINUX_IO_URING_MOCK_FILE_H
#define LINUX_IO_URING_MOCK_FILE_H
#include <linux/types.h>
enum {
IORING_MOCK_FEAT_CMD_COPY,
IORING_MOCK_FEAT_RW_ZERO,
IORING_MOCK_FEAT_RW_NOWAIT,
IORING_MOCK_FEAT_RW_ASYNC,
IORING_MOCK_FEAT_POLL,
IORING_MOCK_FEAT_END,
};
struct io_uring_mock_probe {
__u64 features;
__u64 __resv[9];
};
enum {
IORING_MOCK_CREATE_F_SUPPORT_NOWAIT = 1,
IORING_MOCK_CREATE_F_POLL = 2,
};
struct io_uring_mock_create {
__u32 out_fd;
__u32 flags;
__u64 file_size;
__u64 rw_delay_ns;
__u64 __resv[13];
};
enum {
IORING_MOCK_MGR_CMD_PROBE,
IORING_MOCK_MGR_CMD_CREATE,
};
enum {
IORING_MOCK_CMD_COPY_REGBUF,
};
enum {
IORING_MOCK_COPY_FROM = 1,
};
#endif
