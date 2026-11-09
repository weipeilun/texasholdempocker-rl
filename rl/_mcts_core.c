/*
 * Tree node of the Gumbel AlphaZero search (rl/splendor_mcts.py) in C.
 *
 * Every per-simulation numeric operation of the search (prior normalisation, completed Q,
 * Sequential Halving root selection, interior selection, backup, final policy) on one node
 * object. Results are bit-for-bit identical to the numpy reference (PyNode in
 * rl/splendor_mcts.py): sums use numpy's pairwise summation, operations keep numpy's order,
 * and the module is compiled without floating-point contraction (-ffp-contract=off).
 *
 * Build: python setup_mcts_core.py build_ext --inplace
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

#define N_PLAYERS 4
#define MAX_ACTIONS 256

static const double TINY = 2.2250738585072014e-308;   /* np.finfo(np.float64).tiny */
static const double LOW_LOGIT = -1e9;                   /* mctx score_considered floor */

/* numpy's pairwise summation (numpy/core/src/umath/loops_utils.h.src), unit stride */
static double pairwise_sum(const double *a, Py_ssize_t n)
{
    if (n < 8) {
        double res = 0.;
        for (Py_ssize_t i = 0; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    else if (n <= 128) {
        double r[8], res;
        Py_ssize_t i;
        for (int j = 0; j < 8; j++) {
            r[j] = a[j];
        }
        for (i = 8; i < n - (n % 8); i += 8) {
            r[0] += a[i + 0];
            r[1] += a[i + 1];
            r[2] += a[i + 2];
            r[3] += a[i + 3];
            r[4] += a[i + 4];
            r[5] += a[i + 5];
            r[6] += a[i + 6];
            r[7] += a[i + 7];
        }
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    else {
        Py_ssize_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
    }
}

/* np.argmax: first index of the maximum, a NaN wins */
static int argmax(const double *x, int n)
{
    int best_i = 0;
    double best = x[0];
    if (isnan(best)) {
        return 0;
    }
    for (int i = 1; i < n; i++) {
        if (isnan(x[i])) {
            return i;
        }
        if (x[i] > best) {
            best = x[i];
            best_i = i;
        }
    }
    return best_i;
}

typedef struct {
    PyObject_HEAD
    int n_actions;
    int seat;
    int num_legal;
    long visits;              /* sum of n */
    double *prior;            /* P(s, .), masked and normalised */
    double *logits;           /* log prior, max-normalised over legal actions, -inf for illegal */
    double *n, *w, *q;        /* visit counts, value sums, mean values (from seat's perspective) */
    double *gumbel;           /* root only: gumbel_scale * Gumbel noise, NULL elsewhere */
    unsigned char *mask;
    double value[N_PLAYERS];  /* v_theta(s), absolute seats */
    PyObject *value_tuple;
    PyObject *children;       /* dict: action -> Node | dict (chance node: card -> Node) */
} NodeObject;

static PyTypeObject NodeType;

static NodeObject *node_alloc(PyTypeObject *type, int n_actions)
{
    NodeObject *self = (NodeObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->n_actions = n_actions;
    /* five double arrays + mask in one block */
    size_t doubles = (size_t)5 * n_actions;
    char *block = PyMem_Malloc(doubles * sizeof(double) + (size_t)n_actions);
    if (block == NULL) {
        Py_DECREF(self);
        PyErr_NoMemory();
        return NULL;
    }
    double *d = (double *)block;
    self->prior = d;
    self->logits = d + n_actions;
    self->n = d + 2 * n_actions;
    self->w = d + 3 * n_actions;
    self->q = d + 4 * n_actions;
    self->mask = (unsigned char *)(d + 5 * n_actions);
    memset(self->n, 0, 3 * n_actions * sizeof(double));
    self->gumbel = NULL;
    self->visits = 0;
    self->value_tuple = NULL;
    self->children = PyDict_New();
    if (self->children == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    return self;
}

static void node_dealloc(NodeObject *self)
{
    Py_XDECREF(self->children);
    Py_XDECREF(self->value_tuple);
    if (self->prior != NULL) {
        PyMem_Free(self->prior);
    }
    if (self->gumbel != NULL) {
        PyMem_Free(self->gumbel);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Node(seat, reply, mask)
 *   reply: bytes-like float32 [n_actions action probabilities + N_PLAYERS values relative to seat]
 *   mask:  bytes-like uint8 [n_actions] legal actions */
static PyObject *node_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    int seat;
    Py_buffer reply, mask;
    if (!PyArg_ParseTuple(args, "iy*y*", &seat, &reply, &mask)) {
        return NULL;
    }
    PyObject *result = NULL;
    int n_actions = (int)mask.len;
    if (n_actions < 1 || n_actions > MAX_ACTIONS) {
        PyErr_SetString(PyExc_ValueError, "mask length out of range");
        goto done;
    }
    if (reply.len < (Py_ssize_t)(n_actions + N_PLAYERS) * 4) {
        PyErr_SetString(PyExc_ValueError, "reply too short");
        goto done;
    }
    if (seat < 0 || seat >= N_PLAYERS) {
        PyErr_SetString(PyExc_ValueError, "seat out of range");
        goto done;
    }
    NodeObject *self = node_alloc(type, n_actions);
    if (self == NULL) {
        goto done;
    }
    const unsigned char *m = (const unsigned char *)mask.buf;
    const char *r = (const char *)reply.buf;
    double p[MAX_ACTIONS];
    int num_legal = 0;
    for (int i = 0; i < n_actions; i++) {
        float f;
        memcpy(&f, r + 4 * i, 4);
        self->mask[i] = m[i] ? 1 : 0;
        num_legal += self->mask[i];
        /* np.asarray(probs, float64) * mask */
        p[i] = (double)f * (self->mask[i] ? 1.0 : 0.0);
    }
    self->seat = seat;
    self->num_legal = num_legal;
    double total = pairwise_sum(p, n_actions);
    for (int i = 0; i < n_actions; i++) {
        /* probs / total if total > 0 else mask / mask.sum() */
        self->prior[i] = total > 0 ? p[i] / total : (double)self->mask[i] / (double)num_legal;
    }
    /* logits = where(mask, log(maximum(prior, tiny)), -inf); logits -= logits[mask].max() */
    double lmax = -INFINITY;
    int first = 1;
    for (int i = 0; i < n_actions; i++) {
        double pr = self->prior[i] < TINY ? TINY : self->prior[i];
        double l = log(pr);
        self->logits[i] = l;
        if (self->mask[i] && (first || l > lmax)) {
            lmax = l;
            first = 0;
        }
    }
    for (int i = 0; i < n_actions; i++) {
        self->logits[i] = self->mask[i] ? self->logits[i] - lmax : -INFINITY;
    }
    /* to_absolute(values_rel, seat).astype(float64): value[i] = rel[(i - seat) % N_PLAYERS] */
    for (int i = 0; i < N_PLAYERS; i++) {
        float f;
        memcpy(&f, r + 4 * (n_actions + ((i - seat + N_PLAYERS) % N_PLAYERS)), 4);
        self->value[i] = (double)f;
    }
    self->value_tuple = Py_BuildValue("(dddd)", self->value[0], self->value[1], self->value[2], self->value[3]);
    if (self->value_tuple == NULL) {
        Py_DECREF(self);
        goto done;
    }
    result = (PyObject *)self;
done:
    PyBuffer_Release(&reply);
    PyBuffer_Release(&mask);
    return result;
}

/* sigma(completed Q) of qtransform_completed_by_mix_value; illegal entries are meaningless */
static void completed_q(NodeObject *self, double *out, double maxvisit_init, double value_scale, int rescale)
{
    int A = self->n_actions;
    double raw_value = self->value[self->seat];
    double sum_n = (double)self->visits;
    double mixed;
    double c[MAX_ACTIONS];
    if (sum_n > 0) {
        double pv[MAX_ACTIONS], pq[MAX_ACTIONS];
        int k = 0;
        for (int i = 0; i < A; i++) {
            if (self->n[i] > 0) {
                double pr = self->prior[i] < TINY ? TINY : self->prior[i];
                pv[k] = pr;
                pq[k] = pr * self->q[i];
                k++;
            }
        }
        double sum_probs = pairwise_sum(pv, k);
        double weighted_q = pairwise_sum(pq, k) / sum_probs;
        mixed = (raw_value + sum_n * weighted_q) / (sum_n + 1);
    }
    else {
        mixed = raw_value;
    }
    double nmax = self->n[0];
    for (int i = 0; i < A; i++) {
        c[i] = self->n[i] > 0 ? self->q[i] : mixed;
        if (self->n[i] > nmax) {
            nmax = self->n[i];
        }
    }
    if (rescale) {
        double vmin = INFINITY, vmax = -INFINITY;
        for (int i = 0; i < A; i++) {
            if (self->mask[i]) {
                if (c[i] < vmin) vmin = c[i];
                if (c[i] > vmax) vmax = c[i];
            }
        }
        double d = vmax - vmin;
        double denom = 1e-8 > d ? 1e-8 : d;    /* Python max(d, 1e-8) */
        for (int i = 0; i < A; i++) {
            c[i] = (c[i] - vmin) / denom;
        }
    }
    double s = (maxvisit_init + nmax) * value_scale;
    for (int i = 0; i < A; i++) {
        out[i] = s * c[i];
    }
}

/* _softmax_legal(logits + cq, mask) */
static void softmax_legal(NodeObject *self, const double *cq, double *out)
{
    int A = self->n_actions;
    double x[MAX_ACTIONS], e[MAX_ACTIONS];
    double m = -INFINITY;
    int first = 1;
    for (int i = 0; i < A; i++) {
        x[i] = self->mask[i] ? self->logits[i] + cq[i] : -INFINITY;
        if (self->mask[i] && (first || x[i] > m)) {
            m = x[i];
            first = 0;
        }
    }
    for (int i = 0; i < A; i++) {
        e[i] = self->mask[i] ? exp(x[i] - m) : 0.;
    }
    double total = pairwise_sum(e, A);
    for (int i = 0; i < A; i++) {
        out[i] = e[i] / total;
    }
}

/* argmax of _score_considered: considered = (n == considered_visit) & mask */
static int considered_argmax(NodeObject *self, const double *cq, double considered_visit)
{
    int A = self->n_actions;
    double score[MAX_ACTIONS];
    for (int i = 0; i < A; i++) {
        if (self->mask[i] && self->n[i] == considered_visit) {
            double s = (self->gumbel[i] + self->logits[i]) + cq[i];
            score[i] = s < LOW_LOGIT ? LOW_LOGIT : s;
        }
        else {
            score[i] = -INFINITY;
        }
    }
    return argmax(score, A);
}

static int parse_q_args(PyObject *const *args, double *maxvisit_init, double *value_scale, int *rescale)
{
    *maxvisit_init = PyFloat_AsDouble(args[0]);
    if (*maxvisit_init == -1. && PyErr_Occurred()) return -1;
    *value_scale = PyFloat_AsDouble(args[1]);
    if (*value_scale == -1. && PyErr_Occurred()) return -1;
    *rescale = PyObject_IsTrue(args[2]);
    if (*rescale < 0) return -1;
    return 0;
}

/* select_root(considered_visit, maxvisit_init, value_scale, rescale_values) -> action */
static PyObject *node_select_root(NodeObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "select_root(considered_visit, maxvisit_init, value_scale, rescale_values)");
        return NULL;
    }
    if (self->gumbel == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "set_gumbel() first");
        return NULL;
    }
    double considered_visit = PyFloat_AsDouble(args[0]);
    if (considered_visit == -1. && PyErr_Occurred()) return NULL;
    double mvi, vs;
    int rescale;
    if (parse_q_args(args + 1, &mvi, &vs, &rescale) < 0) return NULL;
    double cq[MAX_ACTIONS];
    completed_q(self, cq, mvi, vs, rescale);
    return PyLong_FromLong(considered_argmax(self, cq, considered_visit));
}

/* select_interior(maxvisit_init, value_scale, rescale_values) -> action
 * argmax over legal actions of improved_policy - n / (1 + sum n) */
static PyObject *node_select_interior(NodeObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "select_interior(maxvisit_init, value_scale, rescale_values)");
        return NULL;
    }
    double mvi, vs;
    int rescale;
    if (parse_q_args(args, &mvi, &vs, &rescale) < 0) return NULL;
    int A = self->n_actions;
    double cq[MAX_ACTIONS], improved[MAX_ACTIONS], score[MAX_ACTIONS];
    completed_q(self, cq, mvi, vs, rescale);
    softmax_legal(self, cq, improved);
    double denom = 1. + (double)self->visits;
    for (int i = 0; i < A; i++) {
        score[i] = self->mask[i] ? improved[i] - self->n[i] / denom : -INFINITY;
    }
    return PyLong_FromLong(argmax(score, A));
}

/* final(maxvisit_init, value_scale, rescale_values) -> (action, policy_target as float64 bytes)
 * action: Sequential Halving's choice among the most visited actions; policy: softmax(logits + sigma(completed q)) */
static PyObject *node_final(NodeObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "final(maxvisit_init, value_scale, rescale_values)");
        return NULL;
    }
    if (self->gumbel == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "set_gumbel() first");
        return NULL;
    }
    double mvi, vs;
    int rescale;
    if (parse_q_args(args, &mvi, &vs, &rescale) < 0) return NULL;
    int A = self->n_actions;
    double cq[MAX_ACTIONS], policy[MAX_ACTIONS];
    completed_q(self, cq, mvi, vs, rescale);
    double nmax = self->n[0];
    for (int i = 1; i < A; i++) {
        if (self->n[i] > nmax) nmax = self->n[i];
    }
    int action = considered_argmax(self, cq, nmax);
    softmax_legal(self, cq, policy);
    PyObject *policy_bytes = PyBytes_FromStringAndSize((const char *)policy, (Py_ssize_t)A * sizeof(double));
    if (policy_bytes == NULL) return NULL;
    return Py_BuildValue("(iN)", action, policy_bytes);
}

/* backup(action, value): value is the backed-up value vector (absolute seats, sequence of floats) */
static PyObject *node_backup(NodeObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "backup(action, value)");
        return NULL;
    }
    long a = PyLong_AsLong(args[0]);
    if (a == -1 && PyErr_Occurred()) return NULL;
    if (a < 0 || a >= self->n_actions) {
        PyErr_SetString(PyExc_IndexError, "action out of range");
        return NULL;
    }
    PyObject *value = args[1];
    double v;
    if (PyTuple_CheckExact(value) && PyTuple_GET_SIZE(value) == N_PLAYERS) {
        v = PyFloat_AsDouble(PyTuple_GET_ITEM(value, self->seat));
    }
    else {
        PyObject *item = PySequence_GetItem(value, self->seat);
        if (item == NULL) return NULL;
        v = PyFloat_AsDouble(item);
        Py_DECREF(item);
    }
    if (v == -1. && PyErr_Occurred()) return NULL;
    self->w[a] += v;
    self->n[a] += 1;
    self->q[a] = self->w[a] / self->n[a];
    self->visits += 1;
    Py_RETURN_NONE;
}

/* set_gumbel(noise): float64 bytes-like [n_actions] (gumbel_scale * Gumbel(0, 1) samples) */
static PyObject *node_set_gumbel(NodeObject *self, PyObject *arg)
{
    Py_buffer buf;
    if (PyObject_GetBuffer(arg, &buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (buf.len != (Py_ssize_t)self->n_actions * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "gumbel must be float64 [n_actions]");
        return NULL;
    }
    if (self->gumbel == NULL) {
        self->gumbel = PyMem_Malloc(self->n_actions * sizeof(double));
        if (self->gumbel == NULL) {
            PyBuffer_Release(&buf);
            return PyErr_NoMemory();
        }
    }
    memcpy(self->gumbel, buf.buf, self->n_actions * sizeof(double));
    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
}

/* clone_fresh(): same prior / mask / value, no visits, no children (a transposition's new tree node) */
static PyObject *node_clone_fresh(NodeObject *self, PyObject *Py_UNUSED(ignored))
{
    NodeObject *c = node_alloc(Py_TYPE(self), self->n_actions);
    if (c == NULL) return NULL;
    int A = self->n_actions;
    memcpy(c->prior, self->prior, A * sizeof(double));
    memcpy(c->logits, self->logits, A * sizeof(double));
    memcpy(c->mask, self->mask, A);
    memcpy(c->value, self->value, sizeof(self->value));
    c->seat = self->seat;
    c->num_legal = self->num_legal;
    Py_INCREF(self->value_tuple);
    c->value_tuple = self->value_tuple;
    return (PyObject *)c;
}

static PyObject *double_list(const double *x, int n)
{
    PyObject *list = PyList_New(n);
    if (list == NULL) return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *f = PyFloat_FromDouble(x[i]);
        if (f == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, f);
    }
    return list;
}

/* stats() -> (n, q, prior, logits) lists, for tests and debugging */
static PyObject *node_stats(NodeObject *self, PyObject *Py_UNUSED(ignored))
{
    int A = self->n_actions;
    return Py_BuildValue("(NNNN)", double_list(self->n, A), double_list(self->q, A),
                         double_list(self->prior, A), double_list(self->logits, A));
}

static PyObject *node_get_seat(NodeObject *self, void *closure) { return PyLong_FromLong(self->seat); }
static PyObject *node_get_num_legal(NodeObject *self, void *closure) { return PyLong_FromLong(self->num_legal); }
static PyObject *node_get_visits(NodeObject *self, void *closure) { return PyLong_FromLong(self->visits); }
static PyObject *node_get_value(NodeObject *self, void *closure) { Py_INCREF(self->value_tuple); return self->value_tuple; }
static PyObject *node_get_children(NodeObject *self, void *closure) { Py_INCREF(self->children); return self->children; }

static PyGetSetDef node_getset[] = {
    {"seat", (getter)node_get_seat, NULL, "acting player of this state", NULL},
    {"num_legal", (getter)node_get_num_legal, NULL, "number of legal actions", NULL},
    {"visits", (getter)node_get_visits, NULL, "number of finished simulations through this node", NULL},
    {"value", (getter)node_get_value, NULL, "v_theta(s) as a tuple, absolute seats", NULL},
    {"children", (getter)node_get_children, NULL, "action -> Node | chance dict (card -> Node)", NULL},
    {NULL}
};

static PyMethodDef node_methods[] = {
    {"select_root", (PyCFunction)(void (*)(void))node_select_root, METH_FASTCALL, NULL},
    {"select_interior", (PyCFunction)(void (*)(void))node_select_interior, METH_FASTCALL, NULL},
    {"final", (PyCFunction)(void (*)(void))node_final, METH_FASTCALL, NULL},
    {"backup", (PyCFunction)(void (*)(void))node_backup, METH_FASTCALL, NULL},
    {"set_gumbel", (PyCFunction)node_set_gumbel, METH_O, NULL},
    {"clone_fresh", (PyCFunction)node_clone_fresh, METH_NOARGS, NULL},
    {"stats", (PyCFunction)node_stats, METH_NOARGS, NULL},
    {NULL}
};

static PyTypeObject NodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "rl._mcts_core.Node",
    .tp_basicsize = sizeof(NodeObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)node_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Gumbel AlphaZero search tree node: Node(seat, reply_float32_bytes, mask_uint8_bytes)",
    .tp_methods = node_methods,
    .tp_getset = node_getset,
    .tp_new = node_new,
};

/* pairwise_sum(float64 buffer) -> float, to check the summation order against numpy */
static PyObject *py_pairwise_sum(PyObject *module, PyObject *arg)
{
    Py_buffer buf;
    if (PyObject_GetBuffer(arg, &buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    double s = pairwise_sum((const double *)buf.buf, buf.len / (Py_ssize_t)sizeof(double));
    PyBuffer_Release(&buf);
    return PyFloat_FromDouble(s);
}

static PyMethodDef module_methods[] = {
    {"pairwise_sum", py_pairwise_sum, METH_O, NULL},
    {NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "_mcts_core", "Gumbel AlphaZero search tree node in C", -1, module_methods
};

PyMODINIT_FUNC PyInit__mcts_core(void)
{
    if (PyType_Ready(&NodeType) < 0) return NULL;
    PyObject *m = PyModule_Create(&module_def);
    if (m == NULL) return NULL;
    Py_INCREF(&NodeType);
    if (PyModule_AddObject(m, "Node", (PyObject *)&NodeType) < 0) {
        Py_DECREF(&NodeType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
