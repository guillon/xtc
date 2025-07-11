/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#ifndef _PERF_EVENT_H
#define _PERF_EVENT_H

#include <stdint.h>

extern int open_perf_events_names(int n_events, const char *names[], int *fds, int *group_fd);
extern int close_perf_events(int n_events, int *fds);
extern void reset_perf_events(int group_fd);
extern void start_perf_events(int group_fd);
extern void stop_perf_events(int group_fd);
extern void read_perf_events(int n_events, const int *fds, uint64_t *results);

#endif
