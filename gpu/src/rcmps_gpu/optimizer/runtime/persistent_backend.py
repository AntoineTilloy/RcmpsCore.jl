"""Isolated cooperative FP64 MMA fusion of one complete DOP853 RK attempt.

All inter-block barriers use cooperative_groups::this_grid().sync(). No CPU
objective, FP32 arithmetic, fast math, or software spin barriers.

PTX fragment mapping follows NVIDIA's m8n8k4 FP64 specification:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
Only the kernel source CUDA remains (the classes that launched it inside the
native conditional graph were removed): persistent_rows and
persistent_rows_cached rewrite it, and persistent_trajectory_backend extracts
the RHS and error reduction of the generated trajectory kernel from the result.
Importing this file does not initialize CUDA.
"""
from __future__ import annotations

CUDA=r'''
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__device__ __forceinline__ double2 zeroz(){ return make_double2(0.,0.); }
__device__ __forceinline__ double2 loadz(const double2* p,int ch,int row,int col,int d){
    return row<d && col<d ? p[ch*d*d+row*d+col] : zeroz();
}
__device__ __forceinline__ double2 gather(const double2* y,int ch,int adj,int row,int col,int d){
    double2 z=loadz(y,ch,adj?col:row,adj?row:col,d); if(adj)z.y=-z.y; return z;
}
__device__ __forceinline__ void mma(double a,double b,double &c0,double &c1){
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+d"(c0),"+d"(c1):"d"(a),"d"(b));
}
__device__ __forceinline__ void zmm(double2 a,double2 b,double &r0,double &r1,double &i0,double &i1){
    mma(a.x,b.x,r0,r1); mma(-a.y,b.y,r0,r1);
    mma(a.x,b.y,i0,i1); mma(a.y,b.x,i0,i1);
}
__device__ double warp_sum(double v){
    for(int shift=16;shift;shift/=2)v+=__shfl_down_sync(0xffffffff,v,shift); return v;
}
extern "C" __global__ void persistent_attempt(
    const double2* y,const double2* f,double2* k,double2* stage,double2* proposed,
    double2* base,double2* temp,const double2* ql,const double2* qr,const double2* rl,
    const double2* rr,const double2* ol,const double2* op_right,const double2* fixed,
    const double* coefficients,const int* indices,const int* adjoints,
    const double2* a,const double2* b,const double* rk_a,const double* rk_b,
    const double* e5,const double* e3,const double* params,double atol,double rtol,
    int d,double* partials,double* norms){
    cg::grid_group grid=cg::this_grid();
    const int lane=threadIdx.x,tiles=(d+7)/8,matrix_blocks=10*tiles*tiles;
    const int scalar=blockIdx.x==matrix_blocks;
    const int channel=blockIdx.x/(tiles*tiles),tile=blockIdx.x%(tiles*tiles);
    const int m=tile/tiles*8,n=tile%tiles*8;
    const int row=m+(lane>>2),col=n+2*(lane&3),total=10*d*d+1;
    int element0=channel*d*d+row*d+col,element1=element0+1;
    const bool valid0=!scalar && row<d && col<d,valid1=!scalar && row<d && col+1<d;
    if(valid0)k[element0]=f[element0]; if(valid1)k[element1]=f[element1];
    if(scalar && lane==0)k[total-1]=f[total-1];
    grid.sync();
    double h=params[0];
    for(int s=1;s<=12;++s){
        const double* weights=s==12?rk_b:rk_a+s*16;
        const int terms=s==12?12:s;
        for(int slot=0;slot<2;++slot){
            int element=slot?element1:element0;
            bool valid=slot?valid1:valid0;
            if(scalar){element=total-1;valid=lane==0 && slot==0;}
            if(valid){
                double2 z=y[element];
                for(int j=0;j<terms;++j){ double2 v=k[j*total+element];
                    z.x+=h*weights[j]*v.x;z.y+=h*weights[j]*v.y; }
                stage[element]=z; if(s==12)proposed[element]=z;
            }
        }
        grid.sync();
        double jac=params[3*s+1],source=params[3*s+2];
        if(!scalar){
            double l0=0.,l1=0.,li0=0.,li1=0.,r0=0.,r1=0.,ri0=0.,ri1=0.;
            double j0=0.,j1=0.,ji0=0.,ji1=0.,il0=0.,il1=0.,ili0=0.,ili1=0.;
            double ir0=0.,ir1=0.,iri0=0.,iri1=0.;
            int ar=m+(lane>>2),bc=n+(lane>>2),kk=lane&3;
            for(int start=0;start<d;start+=4){
                int kp=start+kk;
                zmm(loadz(ql,channel,ar,kp,d),loadz(stage,channel,kp,bc,d),l0,l1,li0,li1);
                zmm(loadz(stage,channel,ar,kp,d),loadz(qr,channel,kp,bc,d),r0,r1,ri0,ri1);
                zmm(loadz(rl,channel,ar,kp,d),gather(stage,indices[channel],adjoints[channel],kp,bc,d),j0,j1,ji0,ji1);
                zmm(loadz(ol,channel,ar,kp,d),gather(stage,indices[10+channel],adjoints[10+channel],kp,bc,d),il0,il1,ili0,ili1);
                zmm(gather(stage,indices[20+channel],adjoints[20+channel],ar,kp,d),loadz(op_right,channel,kp,bc,d),ir0,ir1,iri0,iri1);
            }
            double coefficient=coefficients[channel];
            if(valid0){ double2 v=fixed[element0];
                base[element0]=make_double2(jac*(l0+r0)+source*(coefficient*(il0+ir0)+v.x),
                    jac*(li0+ri0)+source*(coefficient*(ili0+iri0)+v.y));
                temp[element0]=make_double2(j0,ji0); }
            if(valid1){ double2 v=fixed[element1];
                base[element1]=make_double2(jac*(l1+r1)+source*(coefficient*(il1+ir1)+v.x),
                    jac*(li1+ri1)+source*(coefficient*(ili1+iri1)+v.y));
                temp[element1]=make_double2(j1,ji1); }
        }else{
            double trace=0.;
            for(int i=lane;i<d*d;i+=32){ int row=i/d,col=i%d;
                double2 aa=a[i],bb=b[i],v=stage[3*d*d+i],vt=stage[3*d*d+col*d+row];
                trace+=aa.x*v.x+aa.y*v.y+bb.x*vt.x-bb.y*vt.y; }
            trace=warp_sum(trace);if(lane==0)k[s*total+total-1]=make_double2(8.*source*trace,0.);
        }
        grid.sync();
        if(!scalar){
            double r0=0.,r1=0.,i0=0.,i1=0.;int ar=m+(lane>>2),bc=n+(lane>>2),kk=lane&3;
            for(int start=0;start<d;start+=4){int kp=start+kk;
                zmm(loadz(temp,channel,ar,kp,d),loadz(rr,channel,kp,bc,d),r0,r1,i0,i1);}
            if(valid0){double2 v=base[element0];k[s*total+element0]=make_double2(v.x+jac*r0,v.y+jac*i0);}
            if(valid1){double2 v=base[element1];k[s*total+element1]=make_double2(v.x+jac*r1,v.y+jac*i1);}
        }
        grid.sync();
    }
    double sum5=0.,sum3=0.;
    for(int slot=0;slot<2;++slot){
        int element=slot?element1:element0;bool valid=slot?valid1:valid0;
        if(scalar){element=total-1;valid=lane==0 && slot==0;}
        if(valid){double2 z5=zeroz(),z3=zeroz();
            for(int s=0;s<13;++s){double2 z=k[s*total+element];
                z5.x+=e5[s]*z.x;z5.y+=e5[s]*z.y;z3.x+=e3[s]*z.x;z3.y+=e3[s]*z.y;}
            double2 old=y[element],next=proposed[element];
            double scale=atol+rtol*fmax(hypot(old.x,old.y),hypot(next.x,next.y));
            sum5+=(z5.x*z5.x+z5.y*z5.y)/(scale*scale);
            sum3+=(z3.x*z3.x+z3.y*z3.y)/(scale*scale);
        }
    }
    sum5=warp_sum(sum5);sum3=warp_sum(sum3);
    if(lane==0){partials[2*blockIdx.x]=sum5;partials[2*blockIdx.x+1]=sum3;}
    grid.sync();
    if(blockIdx.x==0 && lane==0){double s5=0.,s3=0.;
        for(int b=0;b<gridDim.x;++b){s5+=partials[2*b];s3+=partials[2*b+1];}
        norms[0]=s5;norms[1]=s3;
    }
}
'''
