#ifndef __FIDDLE_H__
#define __FIDDLE_H__

//void pair_stream(int rx, int tx, void* dst, void* src, double size, int type, int step);
void *peer_access(void *addr);
void pair_stream(int rx, int tx, void* dst, void* src, double size, int type);
#endif
