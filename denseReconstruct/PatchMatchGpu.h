#ifndef PATCHMATCH_H
#define PATCHMATCH_H
#include "GlobalState.h"


void pathMatchStereo(int reference,GlobalState& globalState);

#define vecdiv4(v,k) \
v->x = v->x / k; \
v->y = v->y / k; \
v->z = v->z / k;

#define matvecmul4(m, v, out) \
out->x = \
m [0] * v.x + \
m [1] * v.y + \
m [2] * v.z; \
out->y = \
m [3] * v.x + \
m [4] * v.y + \
m [5] * v.z; \
out->z = \
m [6] * v.x + \
m [7] * v.y + \
m [8] * v.z;
#define sub(v0,v1) v0.x = v0.x - v1.x; \
                   v0.y = v0.y - v1.y; \
                   v0.z = v0.z - v1.z;
#define pow2(x) ((x)*(x)) 
#define dot4(v0,v1) v0.x * v1.x + \
                    v0.y * v1.y + \
                    v0.z * v1.z
#define negate4(v) v->x = -v->x; \
                   v->y = -v->y; \
                   v->z = -v->z;  
#define outer_product4(v0,v1, out) \
out[0] = v0.x * v1.x; \
out[1] = v0.x * v1.y; \
out[2] = v0.x * v1.z; \
out[3] = v0.y * v1.x; \
out[4] = v0.y * v1.y; \
out[5] = v0.y * v1.z; \
out[6] = v0.z * v1.x; \
out[7] = v0.z * v1.y; \
out[8] = v0.z * v1.z;
#define matmatsub2(m0, m1) \
m1[0] = m0[0] - m1[0]; \
m1[1] = m0[1] - m1[1]; \
m1[2] = m0[2] - m1[2]; \
m1[3] = m0[3] - m1[3]; \
m1[4] = m0[4] - m1[4]; \
m1[5] = m0[5] - m1[5]; \
m1[6] = m0[6] - m1[6]; \
m1[7] = m0[7] - m1[7]; \
m1[8] = m0[8] - m1[8];


#define matdivide(m,k) \
m[0] = m[0] / k; \
m[1] = m[1] / k; \
m[2] = m[2] / k; \
m[3] = m[3] / k; \
m[4] = m[4] / k; \
m[5] = m[5] / k; \
m[6] = m[6] / k; \
m[7] = m[7] / k; \
m[8] = m[8] / k;

#define matmul_cu(m0, m1, out) \
out[0] = \
m0 [0] * m1[0] + \
m0 [1] * m1[0+3] + \
m0 [2] * m1[0+6]; \
out[1] = \
m0 [0] * m1[1] + \
m0 [1] * m1[1+3] + \
m0 [2] * m1[1+6]; \
out[2] = \
m0 [0] * m1[2] + \
m0 [1] * m1[2+3] + \
m0 [2] * m1[2+6]; \
out[3] = \
m0 [3] * m1[0] + \
m0 [4] * m1[0+3] + \
m0 [5] * m1[0+6]; \
out[4] = \
m0 [3] * m1[1] + \
m0 [4] * m1[1+3] + \
m0 [5] * m1[1+6]; \
out[5] = \
m0 [3] * m1[2] + \
m0 [4] * m1[2+3] + \
m0 [5] * m1[2+6]; \
out[6] = \
m0 [6] * m1[0] + \
m0 [7] * m1[0+3] + \
m0 [8] * m1[0+6]; \
out[7] = \
m0 [6] * m1[1] + \
m0 [7] * m1[1+3] + \
m0 [8] * m1[1+6]; \
out[8] = \
m0 [6] * m1[2] + \
m0 [7] * m1[2+3] + \
m0 [8] * m1[2+6];

#define matvecmul4noz(m, v, out) \
out->x = \
m [0] * v.x + \
m [1] * v.y + \
m [2];\
out->y = \
m [3] * v.x + \
m [4] * v.y + \
m [5]; \
out->z = \
m [6] * v.x + \
m [7] * v.y + \
m [8];



// PatchMatchStereo(GlobalState& globalState);
#endif