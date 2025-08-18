/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <syscall.h>
#include <string.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <alloca.h>
#if HAS_PFM
#include <perfmon/pfmlib.h>
#endif
#include "perf_event.h"

#define PERF_EVENT_CYCLES 0
#define PERF_EVENT_CLOCKS 1
#define PERF_EVENT_INSTRS 2
#define PERF_EVENT_MIGRATIONS 3
#define PERF_EVENT_SWITCHES 4
#define PERF_EVENT_CACHE_ACCESS 5
#define PERF_EVENT_CACHE_MISSES 6
#define PERF_EVENT_BRANCH_INSTRS 7
#define PERF_EVENT_BRANCH_MISSES 8
#define PERF_EVENT_NUM 9

static int all_perf_events[PERF_EVENT_NUM] =
  {
   PERF_EVENT_CYCLES,
   PERF_EVENT_CLOCKS,
   PERF_EVENT_INSTRS,
   PERF_EVENT_MIGRATIONS,
   PERF_EVENT_SWITCHES,
   PERF_EVENT_CACHE_ACCESS,
   PERF_EVENT_CACHE_MISSES,
   PERF_EVENT_BRANCH_INSTRS,
   PERF_EVENT_BRANCH_MISSES,
  };

typedef struct { int id; int type; int num; const char *name; } perf_event_decl_t;

static const perf_event_decl_t perf_events_decl[] =
  {
   { PERF_EVENT_CYCLES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles" },
   { PERF_EVENT_CLOCKS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, "clocks" },
   { PERF_EVENT_INSTRS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions" },
   { PERF_EVENT_MIGRATIONS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, "migrations" },
   { PERF_EVENT_SWITCHES, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "context_switches" },
   { PERF_EVENT_CACHE_ACCESS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, "cache_access" },
   { PERF_EVENT_CACHE_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "cache_misses" },
   { PERF_EVENT_BRANCH_INSTRS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branches" },
   { PERF_EVENT_BRANCH_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branches_misses" },
  };

static __attribute__((constructor)) void perf_event_init(void) {
#if HAS_PFM
    int res = pfm_initialize();
    if (res != PFM_SUCCESS) {
        fprintf(stderr, "ERROR: cannot initialize libpfm: pfm_initialize(): %s\n", pfm_strerror(res));
        exit(EXIT_FAILURE);
    }
#endif /* HAS_PFM */
}

static __attribute__((destructor)) void perf_event_fini(void) {
#if HAS_PFM
    pfm_terminate();
#endif /* HAS_PFM */
}

static int sys_perf_event_open(struct perf_event_attr *hw_event,
                    pid_t pid, int cpu, int group_fd,
                    unsigned long flags)
{
  long fd;
  fd = syscall(SYS_perf_event_open, hw_event, pid, cpu,
	       group_fd, flags);
  return (int)fd;
}

static int get_perf_event_config(const char *name, int *type_ptr, int *event_ptr)
{
    for (int e = 0; e < sizeof(perf_events_decl)/sizeof(*perf_events_decl); e++) {
        if (strcmp(name, perf_events_decl[e].name) == 0) {
            *type_ptr = perf_events_decl[e].type;
            *event_ptr = perf_events_decl[e].num;
            return 0;
        }
    }
    return 1;
}

typedef struct {
    struct perf_event_attr *attr;
    char **fstr;
    size_t size;
    int idx;
    int cpu;
    int flags;
} local_pfm_perf_encode_arg_t;

static int update_perf_event_pfm(const char *name, struct perf_event_attr *attr_ptr)
{
#if HAS_PFM
    local_pfm_perf_encode_arg_t arg;
    memset(&arg, 0, sizeof(arg));
    arg.size = sizeof(arg);
    arg.attr = attr_ptr;
    int ret = pfm_get_os_event_encoding(name, PFM_PLM3, PFM_OS_PERF_EVENT, &arg);
    if (ret != PFM_SUCCESS) {
        return -1;
    }
    return 0;
#else
    return -1;
#endif
}

static int update_perf_event_attr(const char *name, struct perf_event_attr *attr_ptr)
{
    int res, type, config;
    res = get_perf_event_config(name, &type, &config);
    if (res == 0) {
        attr_ptr->type = type;
        attr_ptr->config = config;
        return 0;
    }
    res = update_perf_event_pfm(name, attr_ptr);
    return res;
}

static void init_perf_event_attr(struct perf_event_attr *attr_ptr)
{
  memset(attr_ptr, 0, sizeof(*attr_ptr));
  attr_ptr->size = sizeof(*attr_ptr);
  attr_ptr->exclude_kernel = 1;
  attr_ptr->exclude_hv = 1;
  attr_ptr->exclude_idle = 1;
  attr_ptr->inherit = 1;
  attr_ptr->disabled = 1;
  attr_ptr->read_format = PERF_FORMAT_GROUP;
}

static int open_perf_event_name(const char *name, int group_fd)
{
    int res;
    struct perf_event_attr attr;
    init_perf_event_attr(&attr);
    res = update_perf_event_attr(name, &attr);
    if (res != 0) {
        return -1;
    }
    if (group_fd != -1) {
        attr.disabled = 0;
    }
    return sys_perf_event_open(&attr, 0/*pid*/, -1/*cpu*/,
                               group_fd, 0/*flags*/);
}

int open_perf_events_names(int n_events, const char *names[], int *fds, int *group_fd)
{
    *group_fd = -1;
    for (int i = 0; i < n_events; i++) {
        fds[i] = open_perf_event_name(names[i], *group_fd);
        if (*group_fd == -1 && fds[i] >= 0)
            *group_fd = fds[i];
    }
    if (*group_fd >= 0)
        ioctl(*group_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    return 0;
}

int close_perf_events(int n_events, int *fds)
{
    for (int i = 0; i < n_events; i++) {
        if (fds[i] >= 0)
            close(fds[i]);
    }
    return 0;
}

void reset_perf_events(int group_fd)
{
    if (group_fd < 0)
        return;
    ioctl(group_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
}

void start_perf_events(int group_fd)
{
    if (group_fd < 0)
        return;
    ioctl(group_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

void stop_perf_events(int group_fd)
{
    if (group_fd < 0)
        return;
    ioctl(group_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
}

void read_perf_events(int n_events, const int *fds, uint64_t *results)
{
    int group_fd = -1;
    int events_count = 0;
    for(int i = 0; i < n_events; i++) {
        results[i] = 0;
        if (fds[i] >= 0)
            events_count++;
        if (group_fd == -1 && fds[i] >= 0)
            group_fd = fds[i];
    }
    if (group_fd < 0)
        return;
    uint64_t *values;
    values = alloca((1+events_count)*sizeof(*values));
    ssize_t res = read(group_fd, values, (1+events_count)*sizeof(*values));
    if (res != (1+events_count)*sizeof(*values)) {
        return;
    }
    values++;
    for(int i = 0; i < n_events; i++) {
        if (fds[i] >= 0)
            results[i] = *values++;
    }
}
