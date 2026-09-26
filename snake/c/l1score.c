/*
 * l1score.c - a small local decision model in C: text in, one of N options out, teachable on the fly.
 *
 *   $ printf 'The next gap is above. The bird is falling.\n' | l1score --stdin model.onnx vocab.txt flap.bin
 *   {"label":"flap","p":0.96,"margin":3.29,"tokens":18,"us":7400,"examples":0,"probabilities":{"noop":0.04,"flap":0.96}}
 *
 *   Each input line is one request. TEACH adds a labelled example that takes effect on the next request (each option's logit
 *   also gets DEFAULT_GAMMA x the similarity to its nearest taught example); no retraining, no restart.
 *
 *   state text -> WordPiece (BERT uncased) -> encoder (ONNX Runtime C API), CLS or mean pooling + L2
 *              -> u = s^T (I + D)  ->  logit_c = u . C_c / tau  ->  softmax over the options -> one JSON line
 *
 * The scorer file ("L1S1") carries the trained D, the option names and the precomputed option embeddings
 * C (embedded once with the same encoder), so a request costs one encoder pass plus one dim x dim
 * matrix-vector product. D is trained per standard (leave-one-framework-out showed it does not transfer).
 *
 * Usage: l1score --stdin model.onnx vocab.txt scorer.bin      one JSON line per input line;
 *                                                             "TEACH\t<option>\t<text>" adds a correction, "FORGET" clears them
 *        l1score --tokens vocab.txt scorer.bin "text"          print token ids (tokenizer parity check)
 *
 * File layout (little-endian): "L1S1", u32 dim, u32 n_options, u32 pool (0 mean, 1 cls), u32 max_seq,
 * f32 tau, per option: u8 length (1-64) + ASCII name; f32 D[dim*dim]; f32 C[n*dim].
 * Names are printable ASCII without '"' or '\', so the JSON needs no escaping.
 */

#define _POSIX_C_SOURCE 200809L   /* getline, clock_gettime */

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <onnxruntime_c_api.h>

#define MAX_DIM     1024
#define MAX_OPTS    256
#define MAX_LABEL   64
#define MAX_SEQ     256   /* buffer size; the file's max_seq (<= 256) is what is enforced */
#define MAX_WORD    100   /* BERT: longer words become [UNK] */
#define HASH_SIZE   (1u << 16)   /* > 2x the 30522-entry vocab */

static const OrtApi *ort;

static void die(const char *msg)
{
    fprintf(stderr, "l1score: %s\n", msg);
    exit(1);
}

static void ort_check(OrtStatus *st)
{
    if (st) {
        fprintf(stderr, "l1score: onnxruntime: %s\n", ort->GetErrorMessage(st));
        ort->ReleaseStatus(st);
        exit(1);
    }
}

/* ------------------------------------------------------------------ */
/* Vocabulary: open-addressing hash table, token string -> id          */
/* ------------------------------------------------------------------ */

typedef struct {
    char    *data;              /* whole vocab.txt, newlines -> NUL */
    char    *keys[HASH_SIZE];
    int32_t  ids[HASH_SIZE];
} Vocab;

static uint32_t fnv1a(const char *s, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static int vocab_get(const Vocab *v, const char *s, size_t n)
{
    uint32_t i = fnv1a(s, n) & (HASH_SIZE - 1);
    while (v->keys[i]) {
        if (strlen(v->keys[i]) == n && memcmp(v->keys[i], s, n) == 0)
            return v->ids[i];
        i = (i + 1) & (HASH_SIZE - 1);
    }
    return -1;
}

static Vocab *vocab_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open vocab file");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    Vocab *v = calloc(1, sizeof *v);
    if (!v || !(v->data = malloc((size_t)size + 1))) die("out of memory");
    if (fread(v->data, 1, (size_t)size, f) != (size_t)size) die("cannot read vocab file");
    fclose(f);
    v->data[size] = '\0';

    int32_t id = 0;
    char *line = v->data, *end = v->data + size;
    while (line < end) {
        char *nl = memchr(line, '\n', (size_t)(end - line));
        if (!nl) nl = end;
        *nl = '\0';
        size_t n = (size_t)(nl - line);
        if (n && line[n - 1] == '\r') line[--n] = '\0';

        if ((uint32_t)id >= HASH_SIZE / 2) die("vocab too large");
        uint32_t i = fnv1a(line, n) & (HASH_SIZE - 1);
        while (v->keys[i]) i = (i + 1) & (HASH_SIZE - 1);
        v->keys[i] = line;
        v->ids[i]  = id++;
        line = nl + 1;
    }
    return v;
}

/* ------------------------------------------------------------------ */
/* Tokenizer: BERT-uncased normalisation, basic split, greedy         */
/* WordPiece. Normalisation covers ASCII plus Latin-1/Latin Extended-A */
/* (see fold_cp). Not handled: CJK splitting, non-Latin case folding   */
/* (Greek, Cyrillic), Vietnamese, and Unicode punctuation splitting.   */
/* ------------------------------------------------------------------ */

/* BERT's uncased normaliser is lowercase + NFD + drop combining marks.
 * FOLD is that result for U+00C0..U+017F, generated from Python's
 * unicodedata; '_' marks letters that don't decompose (ß, æ, ø, ...),
 * which are only lowercased, and the symbols × and ÷, which are left alone. */
static const char FOLD[] =
    /* C0 */ "aaaaaa_ceeeeiiii"
    /* D0 */ "_nooooo__uuuuy__"
    /* E0 */ "aaaaaa_ceeeeiiii"
    /* F0 */ "_nooooo__uuuuy_y"
    /* 100 */ "aaaaaaccccccccdd"
    /* 110 */ "__eeeeeeeeeegggg"
    /* 120 */ "gggghh__iiiiiiii"
    /* 130 */ "i___jjkk_llllll_"
    /* 140 */ "___nnnnnn___oooo"
    /* 150 */ "oo__rrrrrrssssss"
    /* 160 */ "sstttt__uuuuuuuu"
    /* 170 */ "uuuuwwyyyzzzzzz_";

/* Normalised form of one codepoint; 0 means "drop it". */
static uint32_t fold_cp(uint32_t cp)
{
    if (cp < 0x80)
        return (uint32_t)tolower((int)cp);
    if (cp >= 0x300 && cp <= 0x36F)          /* combining diacritical marks */
        return 0;
    if (cp >= 0xC0 && cp <= 0x17F) {
        char base = FOLD[cp - 0xC0];
        if (base != '_') return (uint32_t)base;
        switch (cp) {                        /* uppercase, no decomposition */
        case 0xC6: case 0xD0: case 0xD8: case 0xDE:
            return cp + 0x20;                /* Æ Ð Ø Þ */
        case 0x110: case 0x126: case 0x132: case 0x13F:
        case 0x141: case 0x14A: case 0x152: case 0x166:
            return cp + 1;                   /* Đ Ħ Ĳ Ŀ Ł Ŋ Œ Ŧ */
        }
    }
    return cp;
}

#define UTF8_BAD UINT32_MAX

/* Decode one UTF-8 sequence at s. A malformed byte (stray continuation,
 * truncated sequence) returns UTF8_BAD with length 1. */
static uint32_t utf8_next(const unsigned char *s, size_t *len)
{
    unsigned c = s[0];
    size_t   n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    uint32_t cp = n == 4 ? c & 0x07 : n == 3 ? c & 0x0F : n == 2 ? c & 0x1F : c;

    *len = 1;
    if (n == 1) return c < 0x80 ? c : UTF8_BAD;          /* 0x80-0xBF: stray */
    for (size_t i = 1; i < n; i++) {
        if ((s[i] & 0xC0) != 0x80) return UTF8_BAD;      /* also stops at NUL */
        cp = cp << 6 | (s[i] & 0x3F);
    }
    *len = n;
    return cp;
}

static size_t utf8_put(uint32_t cp, char *out)
{
    if (cp < 0x80)  { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | cp >> 6);
                      out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | cp >> 12);
                        out[1] = (char)(0x80 | (cp >> 6 & 0x3F));
                        out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | cp >> 18);
    out[1] = (char)(0x80 | (cp >> 12 & 0x3F));
    out[2] = (char)(0x80 | (cp >> 6 & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Lowercase + strip accents. Output is never longer than the input. */
static char *normalize(const char *text)
{
    size_t len = strlen(text);
    char  *out = malloc(len + 1);
    if (!out) die("out of memory");

    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p;) {
        size_t   n;
        uint32_t cp = utf8_next(p, &n);
        if (cp == UTF8_BAD) out[o++] = (char)*p;     /* keep the raw byte */
        else if ((cp = fold_cp(cp)))  o += utf8_put(cp, out + o);
        p += n;
    }
    out[o] = '\0';
    return out;
}

static size_t wordpiece(const Vocab *v, const char *w, size_t len, int unk,
                        int64_t *out, size_t cap)
{
    if (cap == 0) return 0;
    if (len > MAX_WORD) { out[0] = unk; return 1; }

    char   buf[MAX_WORD + 3] = "##";
    size_t n = 0, start = 0;
    while (start < len) {
        size_t end = len;
        int    id  = -1;
        for (; end > start; end--) {
            if (start == 0) {
                id = vocab_get(v, w, end);
            } else {
                memcpy(buf + 2, w + start, end - start);
                id = vocab_get(v, buf, end - start + 2);
            }
            if (id >= 0) break;
        }
        if (id < 0) { out[0] = unk; return 1; }   /* whole word -> [UNK] */
        if (n < cap) out[n++] = id;
        start = end;
    }
    return n;
}

static size_t tokenize(const Vocab *v, const char *text, size_t max_seq, int64_t *ids)
{
    int cls = vocab_get(v, "[CLS]", 5);
    int sep = vocab_get(v, "[SEP]", 5);
    int unk = vocab_get(v, "[UNK]", 5);
    if (cls < 0 || sep < 0 || unk < 0) die("vocab is missing [CLS]/[SEP]/[UNK]");

    const size_t cap = max_seq - 1;   /* keep room for [SEP] */
    size_t n = 0;
    ids[n++] = cls;

    char  *norm = normalize(text);
    char   word[MAX_WORD + 1];
    size_t wl = 0;       /* true word length; may exceed the buffer */
    for (const unsigned char *p = (const unsigned char *)norm;; p++) {
        int c      = *p;
        int is_sep = c == '\0' || isspace(c);
        int is_pun = c < 0x80 && ispunct(c);

        if ((is_sep || is_pun) && wl) {
            n += wordpiece(v, word, wl, unk, ids + n, cap - n);
            wl = 0;
        }
        if (is_pun) {
            char ch = (char)c;
            n += wordpiece(v, &ch, 1, unk, ids + n, cap - n);
        } else if (!is_sep) {
            if (wl < MAX_WORD) word[wl] = (char)c;
            wl++;
        }
        if (c == '\0' || n >= cap) break;
    }
    free(norm);
    ids[n++] = sep;
    return n;
}

/* ------------------------------------------------------------------ */
/* Scorer: D (dim x dim), option embeddings C (n x dim), temperature   */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t dim, n, pool, max_seq;
    float    tau;
    float   *D, *C, *u, *probs;
    char     labels[MAX_OPTS][MAX_LABEL + 1];
    /* correction memory: labelled state embeddings added at run time by TEACH lines (no retraining, no reload) */
    float   *ex;                 /* n_ex x dim */
    int32_t *exl;                /* option index of each example */
    uint32_t n_ex;
    float    gamma;              /* weight of the nearest-labelled-state term added to each logit */
    float    margin;             /* top logit minus second logit of the last request: survives float saturation of the softmax */
} Scorer;

#define MAX_EXAMPLES 4096
#ifndef DEFAULT_GAMMA
#define DEFAULT_GAMMA 100.0f      /* nearest-example weight; override at build time with -DDEFAULT_GAMMA=... */
#endif

static int label_char_ok(unsigned char c)
{
    return c >= 0x20 && c <= 0x7e && c != '"' && c != '\\';
}

static void read_exact(void *dst, size_t size, size_t count, FILE *f)
{
    if (fread(dst, size, count, f) != count) die("scorer file: truncated");
}

static void scorer_load(Scorer *s, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open scorer file");
    char magic[4];
    read_exact(magic, 1, 4, f);
    if (memcmp(magic, "L1S1", 4) != 0) die("scorer file: bad magic (expected L1S1)");
    uint32_t hdr[4];
    read_exact(hdr, sizeof hdr[0], 4, f);
    read_exact(&s->tau, sizeof s->tau, 1, f);
    s->dim = hdr[0]; s->n = hdr[1]; s->pool = hdr[2]; s->max_seq = hdr[3];
    if (s->dim < 8 || s->dim > MAX_DIM)        die("scorer file: dim out of range");
    if (s->n < 2 || s->n > MAX_OPTS)           die("scorer file: option count must be 2-256");
    if (s->pool > 1)                           die("scorer file: pool must be 0 (mean) or 1 (cls)");
    if (s->max_seq < 8 || s->max_seq > MAX_SEQ) die("scorer file: max_seq must be 8-256");
    if (!(s->tau > 0.0f))                      die("scorer file: tau must be positive");

    for (uint32_t c = 0; c < s->n; c++) {
        unsigned char len;
        read_exact(&len, 1, 1, f);
        if (len == 0 || len > MAX_LABEL) die("scorer file: option name length must be 1-64 bytes");
        read_exact(s->labels[c], 1, len, f);
        s->labels[c][len] = '\0';
        for (size_t i = 0; i < len; i++)
            if (!label_char_ok((unsigned char)s->labels[c][i]))
                die("scorer file: option names must be printable ASCII without '\"' or '\\'");
        for (uint32_t k = 0; k < c; k++)
            if (strcmp(s->labels[k], s->labels[c]) == 0) die("scorer file: duplicate option name");
    }

    s->D     = malloc((size_t)s->dim * s->dim * sizeof *s->D);
    s->C     = malloc((size_t)s->n * s->dim * sizeof *s->C);
    s->u     = malloc(s->dim * sizeof *s->u);
    s->probs = malloc(s->n * sizeof *s->probs);
    s->ex    = malloc((size_t)MAX_EXAMPLES * s->dim * sizeof *s->ex);
    s->exl   = malloc(MAX_EXAMPLES * sizeof *s->exl);
    s->n_ex  = 0;
    s->gamma = DEFAULT_GAMMA;
    if (!s->D || !s->C || !s->u || !s->probs || !s->ex || !s->exl) die("out of memory");
    read_exact(s->D, sizeof *s->D, (size_t)s->dim * s->dim, f);
    read_exact(s->C, sizeof *s->C, (size_t)s->n * s->dim, f);
    if (fgetc(f) != EOF) die("scorer file: unexpected data after the weights");
    fclose(f);
}

static void scorer_free(Scorer *s)
{
    free(s->D); free(s->C); free(s->u); free(s->probs); free(s->ex); free(s->exl);
}

/* Pool the encoder output to one L2-normalised vector. */
static void pool_embed(const Scorer *s, const float *hidden, size_t seq, float *out)
{
    const size_t dim = s->dim;
    if (s->pool == 1) {                                   /* CLS token */
        memcpy(out, hidden, dim * sizeof *out);
    } else {                                              /* mean over all (unpadded) tokens */
        memset(out, 0, dim * sizeof *out);
        for (size_t t = 0; t < seq; t++)
            for (size_t d = 0; d < dim; d++) out[d] += hidden[t * dim + d];
        for (size_t d = 0; d < dim; d++) out[d] /= (float)seq;
    }
    float norm = 0.0f;
    for (size_t d = 0; d < dim; d++) norm += out[d] * out[d];
    norm = sqrtf(norm);
    if (norm < 1e-12f) norm = 1e-12f;
    for (size_t d = 0; d < dim; d++) out[d] /= norm;
}

/* logits_c = (s^T (I + D)) . C_c / tau, then softmax; probabilities left in s->probs. */
static void score(Scorer *s, const float *emb)
{
    const size_t dim = s->dim;
    memcpy(s->u, emb, dim * sizeof *s->u);                /* the identity part of (I + D) */
    for (size_t i = 0; i < dim; i++) {
        const float  e   = emb[i];
        const float *row = s->D + i * dim;
        for (size_t j = 0; j < dim; j++) s->u[j] += e * row[j];
    }
    float nearest[MAX_OPTS];
    for (size_t c = 0; c < s->n; c++) nearest[c] = 0.0f;
    for (uint32_t k = 0; k < s->n_ex; k++) {            /* cosine to each taught state (all vectors are unit length) */
        const float *e = s->ex + (size_t)k * dim;
        float sim = 0.0f;
        for (size_t j = 0; j < dim; j++) sim += emb[j] * e[j];
        if (sim > nearest[s->exl[k]]) nearest[s->exl[k]] = sim;
    }
    float max = -3.0e38f, sum = 0.0f;   /* finite: -ffast-math assumes no infinities */
    float second = -3.0e38f;
    for (size_t c = 0; c < s->n; c++) {
        const float *cv = s->C + c * dim;
        float acc = 0.0f;
        for (size_t j = 0; j < dim; j++) acc += s->u[j] * cv[j];
        s->probs[c] = acc / s->tau + s->gamma * nearest[c];
        if (s->probs[c] > max) { second = max; max = s->probs[c]; } else if (s->probs[c] > second) second = s->probs[c];
    }
    s->margin = max - second;
    for (size_t c = 0; c < s->n; c++) { s->probs[c] = expf(s->probs[c] - max); sum += s->probs[c]; }
    for (size_t c = 0; c < s->n; c++) s->probs[c] /= sum;
}

/* ------------------------------------------------------------------ */
/* ONNX Runtime: encoder session, opened once and reused               */
/* ------------------------------------------------------------------ */

enum { IN_IDS, IN_MASK, IN_TYPES };

typedef struct {
    OrtEnv            *env;
    OrtSessionOptions *opts;
    OrtSession        *session;
    OrtMemoryInfo     *mem;
    OrtAllocator      *alloc;
    size_t             n_in;
    char              *in_names[3];
    int                in_kind[3];
    char              *out_name;     /* output 0: last_hidden_state [1, seq, dim] */
} Encoder;

static void encoder_open(Encoder *e, const char *model_path)
{
    ort_check(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "l1score", &e->env));
    ort_check(ort->CreateSessionOptions(&e->opts));
    ort_check(ort->SetSessionGraphOptimizationLevel(e->opts, ORT_ENABLE_ALL));
    ort_check(ort->SetIntraOpNumThreads(e->opts, 1));
    ort_check(ort->SetInterOpNumThreads(e->opts, 1));
    ort_check(ort->CreateSession(e->env, model_path, e->opts, &e->session));
    ort_check(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &e->mem));
    ort_check(ort->GetAllocatorWithDefaultOptions(&e->alloc));

    ort_check(ort->SessionGetInputCount(e->session, &e->n_in));
    if (e->n_in > 3) die("model has more inputs than expected");
    for (size_t i = 0; i < e->n_in; i++) {
        ort_check(ort->SessionGetInputName(e->session, i, e->alloc, &e->in_names[i]));
        const char *name = e->in_names[i];
        e->in_kind[i] = strcmp(name, "input_ids") == 0      ? IN_IDS
                      : strcmp(name, "attention_mask") == 0 ? IN_MASK
                      : strcmp(name, "token_type_ids") == 0 ? IN_TYPES
                      : -1;
        if (e->in_kind[i] < 0) die("model has an unexpected input name");
    }
    ort_check(ort->SessionGetOutputName(e->session, 0, e->alloc, &e->out_name));
}

static void encoder_close(Encoder *e)
{
    ort_check(ort->AllocatorFree(e->alloc, e->out_name));
    for (size_t i = 0; i < e->n_in; i++)
        ort_check(ort->AllocatorFree(e->alloc, e->in_names[i]));
    ort->ReleaseMemoryInfo(e->mem);
    ort->ReleaseSession(e->session);
    ort->ReleaseSessionOptions(e->opts);
    ort->ReleaseEnv(e->env);
}

static void encoder_embed(const Encoder *e, const Scorer *s, int64_t *ids, size_t seq, float *embedding)
{
    int64_t mask[MAX_SEQ], types[MAX_SEQ];
    for (size_t t = 0; t < seq; t++) { mask[t] = 1; types[t] = 0; }
    int64_t *const data_for[3] = { ids, mask, types };
    const int64_t  shape[2]    = { 1, (int64_t)seq };

    OrtValue *inputs[3];
    for (size_t i = 0; i < e->n_in; i++)
        ort_check(ort->CreateTensorWithDataAsOrtValue(
            e->mem, data_for[e->in_kind[i]], seq * sizeof(int64_t), shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[i]));

    OrtValue *output = NULL;
    ort_check(ort->Run(e->session, NULL, (const char *const *)e->in_names,
                       (const OrtValue *const *)inputs, e->n_in,
                       (const char *const *)&e->out_name, 1, &output));

    OrtTensorTypeAndShapeInfo *info;
    size_t  rank;
    int64_t dims[3];
    ort_check(ort->GetTensorTypeAndShape(output, &info));
    ort_check(ort->GetDimensionsCount(info, &rank));
    if (rank != 3) die("expected a rank-3 last_hidden_state output");
    ort_check(ort->GetDimensions(info, dims, 3));
    ort->ReleaseTensorTypeAndShapeInfo(info);
    if (dims[0] != 1 || dims[1] != (int64_t)seq || dims[2] != (int64_t)s->dim)
        die("unexpected encoder output shape (want [1, seq, dim from the scorer file])");

    float *hidden;
    ort_check(ort->GetTensorMutableData(output, (void **)&hidden));
    pool_embed(s, hidden, seq, embedding);

    ort->ReleaseValue(output);
    for (size_t i = 0; i < e->n_in; i++) ort->ReleaseValue(inputs[i]);
}

/* ------------------------------------------------------------------ */
/* One request: text -> tokens -> embedding -> score -> JSON           */
/* ------------------------------------------------------------------ */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void answer(const Encoder *enc, const Vocab *vocab, Scorer *sc, const char *text)
{
    const double t0 = now_us();
    int64_t ids[MAX_SEQ];
    size_t  seq = tokenize(vocab, text, sc->max_seq, ids);

    float embedding[MAX_DIM];
    encoder_embed(enc, sc, ids, seq, embedding);
    score(sc, embedding);

    size_t best = 0;
    for (size_t c = 1; c < sc->n; c++) if (sc->probs[c] > sc->probs[best]) best = c;
    const double us = now_us() - t0;

    printf("{\"label\":\"%s\",\"p\":%.9g,\"margin\":%.6g,\"tokens\":%zu,\"us\":%.0f,\"examples\":%u,\"probabilities\":{",
           sc->labels[best], sc->probs[best], sc->margin, seq, us, sc->n_ex);
    for (size_t c = 0; c < sc->n; c++)
        printf("%s\"%s\":%.9g", c ? "," : "", sc->labels[c], sc->probs[c]);
    printf("}}\n");
}

/* "TEACH\t<option>\t<state text>": remember this state as an example of that option, effective on the next request.
 * "FORGET": drop all taught examples. Both answer with one JSON line, like every other input line. */
static void teach(const Encoder *enc, const Vocab *vocab, Scorer *sc, const char *arg)
{
    const char *tab = strchr(arg, '\t');
    if (!tab) { printf("{\"error\":\"TEACH needs <option><TAB><text>\"}\n"); return; }
    size_t n = (size_t)(tab - arg);
    int idx = -1;
    for (uint32_t c = 0; c < sc->n; c++)
        if (strlen(sc->labels[c]) == n && memcmp(sc->labels[c], arg, n) == 0) idx = (int)c;
    if (idx < 0)                    { printf("{\"error\":\"unknown option\"}\n"); return; }
    if (sc->n_ex >= MAX_EXAMPLES)   { printf("{\"error\":\"example memory full\"}\n"); return; }
    int64_t ids[MAX_SEQ];
    size_t  seq = tokenize(vocab, tab + 1, sc->max_seq, ids);
    encoder_embed(enc, sc, ids, seq, sc->ex + (size_t)sc->n_ex * sc->dim);
    sc->exl[sc->n_ex++] = idx;
    printf("{\"ok\":true,\"examples\":%u}\n", sc->n_ex);
}

/* ------------------------------------------------------------------ */

static int usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --stdin model.onnx vocab.txt scorer.bin\n"
            "       %s --tokens vocab.txt scorer.bin \"text\"\n", prog, prog);
    return 2;
}

int main(int argc, char **argv)
{
    if (argc == 5 && strcmp(argv[1], "--tokens") == 0) {     /* tokenizer parity check, no model needed */
        static Scorer sc;
        scorer_load(&sc, argv[3]);
        Vocab  *vocab = vocab_load(argv[2]);
        int64_t ids[MAX_SEQ];
        size_t  n = tokenize(vocab, argv[4], sc.max_seq, ids);
        for (size_t i = 0; i < n; i++) printf("%s%lld", i ? " " : "", (long long)ids[i]);
        printf("\n");
        return 0;
    }
    if (argc != 5 || strcmp(argv[1], "--stdin") != 0) return usage(argv[0]);

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) die("onnxruntime library is older than the headers it was built with");

    static Scorer sc;
    scorer_load(&sc, argv[4]);
    Vocab  *vocab = vocab_load(argv[3]);
    Encoder enc;
    encoder_open(&enc, argv[2]);

    char   *line = NULL;
    size_t  cap  = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, stdin)) != -1) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (strncmp(line, "TEACH\t", 6) == 0)   teach(&enc, vocab, &sc, line + 6);
        else if (strcmp(line, "FORGET") == 0)  { sc.n_ex = 0; printf("{\"ok\":true,\"examples\":0}\n"); }
        else                                   answer(&enc, vocab, &sc, line);
        fflush(stdout);
    }
    free(line);
    encoder_close(&enc);
    scorer_free(&sc);
    free(vocab->data);
    free(vocab);
    return 0;
}
