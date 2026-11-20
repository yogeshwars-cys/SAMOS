/*
 * samos_engine.c -- SAMOS secondary-side retrieval engine.
 *
 * Storage-native: serves queries directly from a SAMOSDB1 store file
 * (memory-mapped on POSIX, buffered on Windows). The store is the only
 * state; there is no separate index to keep in sync -- IVF
 * lists, records and doc ids all live in the one container.
 *
 * RAM offload modes:
 *   --mmap  serve from the file mapping (OS page cache is the cache;
 *           page faults are measured and reported, never hidden)
 *   --ram   copy records into heap RAM, touch every page, and try
 *           mlock() so the working set stays resident; locked bytes
 *           and faults are reported in every response.
 *
 * Canonical semantics (must match samos_common.reference_search):
 *   exact int8 dot over the selected lists, ordered by
 *   (score DESC, idx ASC). Ties are never arrival-ordered.
 *
 * Build (Android): aarch64-linux-android31-clang -O3 -fPIE -pie \
 *     -o samos_engine samos_engine.c
 * Build (host):    gcc -O2 -o samos_engine samos_engine.c -lws2_32
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
static int64_t now_us(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (int64_t)(c.QuadPart*1000000LL/f.QuadPart); }
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
typedef int sock_t;
#define CLOSESOCK close
static int64_t now_us(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (int64_t)ts.tv_sec*1000000LL + ts.tv_nsec/1000; }
#endif

#define SAMO_MAGIC 0x4F4D4153U /* "SAMO" */
#define DIM 384
#define REC_BYTES 512
#define PAD_WORDS 6
#define MAX_TOPK 64
#define MAX_LISTS 4096

struct store {
    uint8_t *base; size_t len;
    uint32_t n, nlist;
    uint8_t *records, *lists_sec, *list_data, *docids_sec, *crc_sec;
    float *centroids;
    uint32_t n_listed;
    uint64_t *doc_tab; char *doc_blob;
    char *manifest; uint32_t manifest_len;
#ifndef _WIN32
    int fd;
#endif
};

static uint32_t crc_table[256]; static int crc_ready=0;
static uint32_t crc32_of(const uint8_t *p, size_t n){
    if(!crc_ready){ for(uint32_t i=0;i<256;i++){ uint32_t c=i; for(int k=0;k<8;k++) c = (c&1)? 0xEDB88320U^(c>>1) : c>>1; crc_table[i]=c; } crc_ready=1; }
    uint32_t c=0xFFFFFFFFU; for(size_t i=0;i<n;i++) c = crc_table[(c^p[i])&0xFF] ^ (c>>8); return c^0xFFFFFFFFU;
}
static uint32_t rd32(const uint8_t*p){ uint32_t v; memcpy(&v,p,4); return v; }
static uint64_t rd64(const uint8_t*p){ uint64_t v; memcpy(&v,p,8); return v; }

static int load_store(struct store *s, const char *path){
    memset(s,0,sizeof(*s));
#ifdef _WIN32
    FILE *f=fopen(path,"rb"); if(!f){ perror("open store"); return -1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    s->base=(uint8_t*)malloc((size_t)sz); if(!s->base){ fclose(f); return -1; }
    if(fread(s->base,1,(size_t)sz,f)!=(size_t)sz){ fclose(f); return -1; }
    fclose(f); s->len=(size_t)sz;
#else
    s->fd=open(path,O_RDONLY); if(s->fd<0){ perror("open store"); return -1; }
    struct stat st; if(fstat(s->fd,&st)!=0){ perror("stat store"); return -1; }
    s->len=(size_t)st.st_size;
    void *m=mmap(NULL,s->len,PROT_READ,MAP_SHARED,s->fd,0);
    if(m==MAP_FAILED){ perror("mmap store"); return -1; }
    s->base=(uint8_t*)m;
#endif
    if(s->len<128 || memcmp(s->base,"SAMOSDB1",8)!=0){ fprintf(stderr,"bad SAMOS magic\n"); return -1; }
    if(rd32(s->base+8)!=2){ fprintf(stderr,"unsupported store version (need v2)\n"); return -1; }
    s->n=rd32(s->base+24); s->nlist=rd32(s->base+32);
    /* n is u64 in header; stores are < 2^32 records by design */
    uint64_t off_rec=rd64(s->base+40), off_cent=rd64(s->base+56),
             off_lists=rd64(s->base+64), off_ld=rd64(s->base+72), off_doc=rd64(s->base+80),
             off_crc=rd64(s->base+88), off_man=rd64(s->base+104);
    s->manifest_len=rd32(s->base+96);
    s->records=s->base+off_rec;
    s->centroids=(float*)(s->base+off_cent);
    s->lists_sec=s->base+off_lists; s->list_data=s->base+off_ld;
    s->docids_sec=s->base+off_doc; s->crc_sec=s->base+off_crc;
    s->manifest=(char*)(s->base+off_man);
    s->doc_tab=(uint64_t*)s->docids_sec;
    s->doc_blob=(char*)(s->docids_sec + (size_t)(s->n+1)*8);
    s->n_listed=0; for(uint32_t i=0;i<s->nlist;i++) s->n_listed += rd32(s->lists_sec + (size_t)i*12 + 8);
    return 0;
}

static int verify_store(struct store *s){
    int bad=0;
    for(uint32_t i=0;i<s->n;i++){
        uint32_t want=rd32(s->crc_sec + (size_t)i*4);
        if(crc32_of(s->records + (size_t)i*REC_BYTES, REC_BYTES)!=want) bad++;
    }
    return bad;
}

static uint32_t rss_kb(void){
#ifdef _WIN32
    return 0;
#else
    FILE *f=fopen("/proc/self/status","r"); if(!f) return 0;
    char line[256]; uint32_t v=0;
    while(fgets(line,sizeof(line),f)){ if(sscanf(line,"VmRSS: %u",&v)==1) break; }
    fclose(f); return v;
#endif
}
static long minflt(void){
#ifndef _WIN32
    struct rusage ru; getrusage(RUSAGE_SELF,&ru); return ru.ru_minflt;
#else
    return 0;
#endif
}

/* RAM-offloaded working set (optional) */
static uint8_t *g_ram_rec=NULL; static size_t g_locked=0;
static int ram_stage(struct store *s){
    size_t rb=(size_t)s->n*REC_BYTES;
    g_ram_rec=(uint8_t*)malloc(rb);
    if(!g_ram_rec){ fprintf(stderr,"ram alloc failed\n"); return -1; }
    memcpy(g_ram_rec,s->records,rb);
    volatile uint8_t t=0; for(size_t i=0;i<rb;i+=4096) t^=g_ram_rec[i]; (void)t;
#ifndef _WIN32
    if(mlock(g_ram_rec,rb)==0) g_locked+=rb; else fprintf(stderr,"mlock records: %s\n",strerror(errno));
#endif
    s->records=g_ram_rec;
    return 0;
}
/* Runtime arena: writable secondary RAM segments (Fabric bus).
   Segment size is fixed at 1 MiB; a segment id is a coordinate the
   primary computed when it declared the segment. */
#define SEG_BYTES (1024u*1024u)
static uint8_t *g_arena=NULL; static uint32_t g_nsegs=0;
static int arena_stage(int arena_mb){
    g_nsegs=(uint32_t)arena_mb;
    size_t bytes=(size_t)g_nsegs*SEG_BYTES;
    g_arena=(uint8_t*)malloc(bytes);
    if(!g_arena){ fprintf(stderr,"arena alloc failed\n"); return -1; }
    volatile uint8_t t=0; for(size_t i=0;i<bytes;i+=4096) t^=g_arena[i]; (void)t;
#ifndef _WIN32
    { size_t done=0; const size_t step=8u*1024u*1024u;
      while(done<bytes){ size_t want=bytes-done<step?bytes-done:step;
        if(mlock(g_arena+done,want)==0) g_locked+=want; else break; done+=want; }
      if(g_locked<(size_t)arena_mb*SEG_BYTES)
        fprintf(stderr,"mlock arena: locked %llu of %llu B (limit)\n",
                (unsigned long long)g_locked,(unsigned long long)bytes); }
#endif
    return 0;
}

struct scored { float score; uint32_t idx; uint32_t ham; };
static int cmp_scored(const void *a, const void *b){
    const struct scored *x=(const struct scored*)a, *y=(const struct scored*)b;
    if(x->score!=y->score) return x->score>y->score?-1:1;
    return x->idx<y->idx?-1:(x->idx>y->idx?1:0);
}

static int read_exact(sock_t fd, void *buf, size_t len){
    size_t tot=0; while(tot<len){ 
#ifdef _WIN32
        int n=recv(fd,(char*)buf+tot,(int)(len-tot),0);
#else
        ssize_t n=read(fd,(char*)buf+tot,len-tot);
#endif
        if(n<=0) return -1; tot+=(size_t)n; } return 0;
}
static int write_exact(sock_t fd, const void *buf, size_t len){
    size_t tot=0; while(tot<len){
#ifdef _WIN32
        int n=send(fd,(const char*)buf+tot,(int)(len-tot),0);
#else
        ssize_t n=write(fd,(const char*)buf+tot,len-tot);
#endif
        if(n<=0) return -1; tot+=(size_t)n; } return 0;
}

struct req_hdr {
    uint32_t magic; uint16_t cmd; uint16_t flags;
    uint32_t nprobe; uint32_t topk; uint32_t rerank_m; float q_scale;
    uint64_t q_pad[PAD_WORDS]; int8_t q_i8[DIM];
};
struct resp_hdr {
    uint32_t magic; uint32_t status; uint32_t n_results; uint32_t scanned;
    uint64_t scan_us; uint64_t rerank_us; uint32_t rss_kb; uint32_t faults;
    uint32_t n_records; uint32_t unique_records; uint64_t locked_bytes; uint32_t mode_ram;
};
struct result { uint32_t idx; float score; uint32_t ham; char docid[32]; };

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: samos_engine <store.samos> [--port N] [--ram|--mmap] [--verify]\n"); return 2; }
    const char *path=argv[1]; int port=9977; int use_ram=0; int verify=0;
    int arena_mb=0;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--ram")) use_ram=1;
        else if(!strcmp(argv[i],"--mmap")) use_ram=0;
        else if(!strcmp(argv[i],"--verify")) verify=1;
        else if(!strcmp(argv[i],"--port")&&i+1<argc) port=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--arena")&&i+1<argc) arena_mb=atoi(argv[++i]);
    }
#ifdef _WIN32
    WSADATA w; WSAStartup(MAKEWORD(2,2),&w);
#endif
    struct store s; if(load_store(&s,path)!=0) return 1;
    if(verify){ int bad=verify_store(&s); printf("crc verify: %d corrupt of %u records\n",bad,s.n); if(bad) return 3; }
    if(use_ram && ram_stage(&s)!=0) return 1;
    if(arena_mb>0 && arena_stage(arena_mb)!=0) return 1;
    printf("samos_engine: %u records, %u lists, mode=%s locked=%lluB rss=%uKB store=%s\n",
        s.n, s.nlist, use_ram?"RAM":"MMAP", (unsigned long long)g_locked, rss_kb(), path);
    fflush(stdout);

    sock_t srv=socket(AF_INET,SOCK_STREAM,0);
    int opt=1; setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,(const char*)&opt,sizeof(opt));
    struct sockaddr_in addr; memset(&addr,0,sizeof(addr));
    addr.sin_family=AF_INET; addr.sin_port=htons((uint16_t)port); addr.sin_addr.s_addr=htonl(INADDR_ANY);
    if(bind(srv,(struct sockaddr*)&addr,sizeof(addr))!=0){ perror("bind"); return 1; }
    listen(srv,4);

    while(1){
        sock_t cli=accept(srv,NULL,NULL); if((int)cli<0) continue;
        int flag=1; setsockopt(cli,IPPROTO_TCP,TCP_NODELAY,(const char*)&flag,sizeof(flag));
        while(1){
            struct req_hdr req;
            if(read_exact(cli,&req,sizeof(req))!=0) break;
            if(req.magic!=SAMO_MAGIC) break;
            struct resp_hdr resp; memset(&resp,0,sizeof(resp));
            resp.magic=SAMO_MAGIC; resp.n_records=s.n; resp.unique_records=s.n;
            resp.locked_bytes=g_locked; resp.mode_ram=use_ram?1:0; resp.rss_kb=rss_kb();
            if(req.cmd==0){ /* HELLO: header + manifest */
                resp.status=0;
                if(write_exact(cli,&resp,sizeof(resp))!=0) break;
                uint32_t ml=s.manifest_len;
                if(write_exact(cli,&ml,4)!=0) break;
                if(write_exact(cli,s.manifest,ml)!=0) break;
                continue;
            }
            if(req.cmd==3 || req.cmd==4){ /* Fabric bus: segment write/read.
                               nprobe=segment, topk=offset, rerank_m=len. */
                uint32_t seg=req.nprobe, off=req.topk, len=req.rerank_m;
                if(!g_arena || seg>=g_nsegs || (uint64_t)off+len>SEG_BYTES){
                    resp.status=3; if(write_exact(cli,&resp,sizeof(resp))!=0) break; continue;
                }
                uint8_t *dst=g_arena + (size_t)seg*SEG_BYTES + off;
                if(req.cmd==3){
                    if(read_exact(cli,dst,len)!=0) break;
                    resp.status=0; resp.n_results=0; resp.scanned=len; resp.rss_kb=rss_kb();
                    if(write_exact(cli,&resp,sizeof(resp))!=0) break;
                } else {
                    long f0=minflt();
                    resp.status=0; resp.n_results=0; resp.scanned=len;
                    resp.faults=(uint32_t)(minflt()-f0); resp.rss_kb=rss_kb();
                    if(write_exact(cli,&resp,sizeof(resp))!=0) break;
                    if(write_exact(cli,dst,len)!=0) break;
                }
                continue;
            }
            if(req.cmd==2){ /* FETCH: direct translated read, no search.
                               req.nprobe carries the record index (the
                               coordinate the primary computed at push). */
                uint32_t idx=req.nprobe;
                if(idx>=s.n){ resp.status=2; if(write_exact(cli,&resp,sizeof(resp))!=0) break; continue; }
                long f0=minflt(); int64_t t0=now_us();
                const uint8_t *rec=s.records + (size_t)idx*REC_BYTES;
                struct result r; memset(&r,0,sizeof(r));
                r.idx=idx; r.score=0.0f; r.ham=0;
                uint64_t a=s.doc_tab[idx], b=s.doc_tab[idx+1];
                size_t dl=(size_t)(b-a); if(dl>31) dl=31;
                memcpy(r.docid, s.doc_blob + a, dl);
                int64_t t1=now_us();
                resp.status=0; resp.n_results=1; resp.scanned=0;
                resp.scan_us=(uint64_t)(t1-t0); resp.rerank_us=0;
                resp.faults=(uint32_t)(minflt()-f0); resp.rss_kb=rss_kb();
                if(write_exact(cli,&resp,sizeof(resp))!=0) break;
                if(write_exact(cli,&r,sizeof(r))!=0) break;
                if(req.flags & 1){ if(write_exact(cli,rec,REC_BYTES)!=0) break; }
                continue;
            }
            if(req.cmd!=1) break;
            uint32_t nprobe=req.nprobe, topk=req.topk;
            if(topk==0||topk>MAX_TOPK) topk=10;
            uint32_t lists[MAX_LISTS];
            if(nprobe>s.nlist) nprobe=s.nlist;
            if(nprobe>MAX_LISTS) break;
            if(read_exact(cli,lists,(size_t)nprobe*4)!=0) break;
            long f0=minflt();
            int64_t t0=now_us();
            /* Exact INT8 dot over every record in the selected lists.
               No approximate prefilter stage. */
            struct scored *tmp=(struct scored*)malloc(sizeof(struct scored)*(s.n_listed?s.n_listed:1));
            uint32_t nc=0; uint32_t scanned=0;
            for(uint32_t li=0; li<nprobe; li++){
                uint32_t lid=lists[li]; if(lid>=s.nlist) continue;
                uint64_t loff=rd64(s.lists_sec+(size_t)lid*12); uint32_t cnt=rd32(s.lists_sec+(size_t)lid*12+8);
                const uint32_t *mem=(const uint32_t*)(s.list_data + (size_t)loff*4);
                scanned+=cnt;
                for(uint32_t j=0;j<cnt;j++){
                    uint32_t idx=mem[j];
                    const uint8_t *rec=s.records + (size_t)idx*REC_BYTES;
                    float dscale; memcpy(&dscale,rec,4);
                    const int8_t *dv=(const int8_t*)(rec+4);
                    int32_t acc=0; for(int d=0;d<DIM;d++) acc += (int32_t)dv[d]*(int32_t)req.q_i8[d];
                    float sc1 = dscale * req.q_scale; float score = sc1 * (float)acc;
                    tmp[nc].score=score; tmp[nc].idx=idx; tmp[nc].ham=0; nc++;
                }
            }
            int64_t t1=now_us();
            qsort(tmp,nc,sizeof(struct scored),cmp_scored);
            uint32_t nout = nc<topk? nc : topk;
            int64_t t2=now_us();
            resp.status=0; resp.n_results=nout; resp.scanned=scanned;
            resp.scan_us=(uint64_t)(t1-t0); resp.rerank_us=(uint64_t)(t2-t1);
            resp.faults=(uint32_t)(minflt()-f0); resp.rss_kb=rss_kb();
            if(write_exact(cli,&resp,sizeof(resp))!=0) break;
            int failed=0;
            for(uint32_t k=0;k<nout;k++){
                struct result r; memset(&r,0,sizeof(r));
                r.idx=tmp[k].idx; r.score=tmp[k].score; r.ham=tmp[k].ham;
                uint32_t idx=tmp[k].idx;
                uint64_t a=s.doc_tab[idx], b=s.doc_tab[idx+1];
                size_t dl=(size_t)(b-a); if(dl>31) dl=31;
                memcpy(r.docid, s.doc_blob + a, dl);
                if(write_exact(cli,&r,sizeof(r))!=0){ failed=1; break; }
                if(req.flags & 1){
                    const uint8_t *rec=s.records + (size_t)idx*REC_BYTES + 388;
                    if(write_exact(cli,rec,124)!=0){ failed=1; break; }
                }
            }
            free(tmp);
            if(failed) break;
        }
        CLOSESOCK(cli);
    }
    return 0;
}
