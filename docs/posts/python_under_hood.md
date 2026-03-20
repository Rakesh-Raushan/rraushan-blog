---
title: Python Remembers Everything — Until You Ask It to Forget
date: 2026-03-20
draft: false
authors:
  - rraushan
tags:
  - python
---

Python's memory model for ML engineers

<!-- more -->

# Python Remembers Everything — Until You Ask It to Forget

A deep dive into Python's memory model for ML engineers who've been surprised by OOM

## TL;DR

An OOM issue in a containerised pipeline turned out to be caused by subtle reference cycles from Python’s import system and lazy loading. A single `gc.collect()` fixed it—but validating that fix made it clear that a solid grasp of Python’s memory model (RSS, refcounts, arenas, GC cycles) isn’t optional for serious ML engineers.

---

> *Don't fear the garbage collector. Don't love it too much. Understand it and use it wisely.*

---

Before We Begin (Motivation)

Debugging memory issues teaches you to read graphs like signals, most spikes are noise, but some patterns matter. This case looked ordinary at first: a containerised inference pipeline that ran fine once, but reliably failed with an OOM kill on the second concurrent run.

Initial suspects like model session cache holding onto arrays or onnx inferencer memory leaks didn’t pan out. Memory measurements showed minimal retained memory, yet the container still died. After digging deeper into RSS, references, and execution flow, the issue turned out not to be in our code at all.

The root cause was an interaction between Python’s import system, a medical library’s lazy loading, and dynamic class construction—creating subtle reference cycles that retained large amounts of memory longer than expected.

*`Three reasonable design decisions. One unexpected outcome.`*

The fix was a single line, an explicit `gc.collect()`. But being confident it wasn’t hiding a deeper issue required understanding Python’s memory model far more deeply than expected; and that’s what this article is about.

*By the end, you’ll have the mental model and practical intuition needed to confidently navigate and debug Python memory behaviour in real-world systems. This is a long read, but I’d recommend staying with it—the pieces only click fully when seen together.*

---

## A Map of Where We Are Going

We are going to cover a lot of ground. Python's `reference counting system` and why it handles most objects instantly. The `cyclic garbage collector` and why it does not. `Memory arenas` — the reason your RSS number lies to you after you free a large array. `Frame objects` — a Python concept most engineers have never thought about, and the actual mechanism behind some of the most confusing memory behaviour in ML pipelines. Lazy imports and how third-party libraries can create reference cycles in code you never wrote. And finally, `gc.collect()` — what it actually does, when it is the right answer, and how to make sure you know the difference.

**Parts 1–5** build the mental model. They build on each other deliberately — each part explains something the next depends on. If you are tempted to skip ahead to the investigation, I'd encourage you not to. The investigation is only satisfying if you can follow the reasoning at each step.

**Part 6** is the investigation itself — the pipeline that kept dying, walked through step by step.

**Part 7** is a practical toolkit reference to return to during your own debugging sessions.

**Part 8** is short. It says the thing the whole article is really about.

One note on scope: everything here applies to **CPython** — the standard Python implementation you almost certainly run in production. PyPy and other implementations have different memory models.

Let's start at the beginning.

---

## Part 1: How Python Actually Manages Memory

*Most engineers treat Python's memory model as a black box. That's fine — until it isn't.*

---

### 1.1 Reference Counting: The Primary Mechanism

Every object in Python has a hidden number attached to it. You never see it. You never set it. But Python is updating it constantly, millions of times per second in a busy program. This number is the **reference count** — and it is the foundation of almost everything Python does with memory.

The rule is simple: every time something points to an object, its reference count goes up by one. Every time that pointer goes away, the count drops by one. The moment the count hits zero, Python frees the object immediately — right then, inline, no waiting.

```python
import sys

x = [1, 2, 3]
print(sys.getrefcount(x))  # 2 — x itself, plus getrefcount's argument

y = x                       # y now also points to the same list
print(sys.getrefcount(x))  # 3

del y                       # y is gone, count drops
print(sys.getrefcount(x))  # 2 again
```

> `getrefcount` always shows one extra because passing the object to the function is itself a reference.

When you write `del x`, you are not deleting the object — you are removing one name that pointed to it. If something else still points to it, the object stays alive. If nothing does, it is freed.

This is why Python feels so different from languages with garbage collectors that pause your program. Most of the time, memory is reclaimed *the instant* it is no longer needed — no pause, no delay, no separate GC thread.

**In ML terms:** when your inference function returns and the local variable holding a 500 MB output array goes out of scope, Python decrements that array's refcount. If nothing else holds a reference to it, it is freed immediately. This is the happy path — and most of the time, it works exactly as you would expect.

---

### 1.2 The Refcount Limitation: Cycles

Here is where the elegant simplicity hits a wall.

What happens when two objects point to each other?

```python
a = []
b = []

a.append(b)   # a holds a reference to b
b.append(a)   # b holds a reference to a

del a
del b
# Both objects are now unreachable from your code.
# But both still have refcount = 1 because they reference each other.
# Refcount GC cannot free them.
```

After `del a` and `del b`, neither object is reachable from your code. But Python's refcount mechanism cannot see that — each object still has a count of 1 because the other is pointing at it. From the refcount's perspective, they are both still alive.

This is a **reference cycle**, and it is the fundamental limitation of refcount-based memory management.

In simple scripts, cycles are rare and the consequences are minor. In a production ML pipeline running large models on large arrays, a cycle that anchors the wrong object at the wrong time is the difference between a successful second run and an OOM-killed container.

---

### 1.3 Cyclic GC: The Safety Net

Python's answer to this problem is a second, separate garbage collector. This is what people usually mean when they say "Python's garbage collector" — but it is actually the *secondary* mechanism, not the primary one.

The cyclic GC works differently from refcounting. Instead of tracking every pointer change, it periodically scans groups of objects, looks for isolated cycles, and frees them.

```python
import gc

freed = gc.collect()
print(f"Freed {freed} objects in cycles")
```

You can trigger it manually, or let Python trigger it automatically based on how many objects have been allocated since the last run. It does not run on a fixed timer. It does not run after every function call. **It runs when Python decides conditions are right — which could be immediately, or could be many seconds later.**

This is the key implication: objects caught in reference cycles are not freed immediately when they become unreachable. They wait until the cyclic GC runs. In a tight inference loop allocating and freeing hundreds of megabytes per task, "waiting" can mean the next task starts before the previous task's memory is released — and the combined footprint exceeds your container limit.

---

### 1.4 The Generational Model: Why Some Objects Linger Longer

Python's cyclic GC uses a **generational** model, dividing objects into three generations based on how long they have survived.

The intuition is simple and clever: most objects die young. A local variable in a function, a temporary list in a comprehension, an intermediate tensor — these typically live for milliseconds. A loaded model, a session cache — these live for the lifetime of the process. Python checks Generation 0 (youngest) most frequently, and Generation 2 (oldest) rarely.

```python
import gc

print(gc.get_threshold())  # (700, 10, 10) by default
# Gen 0 collected after 700 net allocations
# Gen 1 collected after Gen 0 has been collected 10 times
# Gen 2 collected after Gen 1 has been collected 10 times

print(gc.get_count())      # (312, 4, 1) — current counts toward each threshold
```

**The practical implication for ML:** your inference output arrays are Generation 0 objects. They should be collected quickly. But if they get caught in a reference cycle, they survive into Generation 1, then Generation 2, and by that point Python may not collect them for a long time. The array *looks* long-lived because its refcount never hit zero — the generational model is doing exactly what it was designed to do.

---

### 1.5 The Mental Model So Far

```
Object created
      │
      ▼
Refcount > 0 ──── refcount drops to 0 ──► Freed immediately ✓
      │
      │ (caught in a cycle — refcount never hits zero)
      ▼
Waits for cyclic GC
      │
      ├── Gen 0 threshold hit ──► collected quickly
      │
      └── Survived Gen 0 ──► promoted to Gen 1, then Gen 2
                               collected much less frequently
```

To carry forward:
- `del x` removes a name. The object is freed only if nothing else points to it.
- Objects in cycles wait for cyclic GC. This is periodic and unpredictable.
- Cycles that survive long enough get promoted and wait even longer.
- In ML workloads, the objects most likely to be in cycles are also the largest ones.

---

## Part 2: Memory Arenas — Why RSS Lies to You

*You freed the array. RSS didn't move. You're not leaking — you're just looking at the wrong number.*

---

### 2.1 The Number That Feels Like a Lie

You have just run inference. You delete the output array. You call `del output`. You check your memory usage. The number has not moved.

Your first instinct is panic. *Did it not free? Is there a leak? Why is RSS still at 2 GB?*

Most of the time, nothing is wrong. You are looking at the wrong metric, measuring the wrong thing, and drawing the wrong conclusion. To understand why, we need to talk about **memory arenas**.

---

### 2.2 The Cost of Asking the OS for Memory

Every time your program needs memory, it could ask the operating system directly — a `malloc` call that goes all the way to the kernel. In theory, clean. In practice, catastrophically slow for programs that allocate frequently.

System calls are expensive. Python itself creates and destroys objects constantly: every integer, every string, every list element, every function call frame. Making a kernel call for each one would make Python unusably slow.

The solution — used by Python, numpy, ONNX Runtime, and virtually every performance-sensitive runtime — is to **request memory from the OS in large chunks and manage it internally**. These chunks are called arenas.

---

### 2.3 How Arenas Work

Instead of asking the OS for exactly as much memory as you need each time, the allocator asks for a large block upfront. This block is the arena. The allocator then subdivides it internally to satisfy individual allocations, without going back to the OS each time.

When an object inside the arena is freed, the memory goes back into the arena's internal pool — available for the next allocation — but **the arena itself stays resident**. The OS does not know or care that the memory inside it is now "free."

```
┌─────────────────────────────────────────┐
│              Arena (e.g. 256 KB)        │
│                                         │
│  [obj A — in use][obj B — freed!][obj C]│
│                  ^^^^^^^^^^^            │
│                  Available internally   │
│                  but NOT returned to OS │
└─────────────────────────────────────────┘
        OS still counts this as 256 KB used
```

RSS — Resident Set Size, the number you see in `top`, `docker stats`, or `/proc/self/status` — measures how much physical memory the OS has mapped to your process. It has no visibility into what is happening inside an arena. Freed objects inside an arena do not reduce RSS.

**This is not a bug. It is the intended behaviour.** Keeping the arena resident means the next allocation is fast — the memory is already there, already mapped, no kernel call needed.

---

### 2.4 Multiple Arena Systems in an ML Workload

There are actually multiple arena systems at play in a typical ML pipeline, each operating independently.

**Python's own allocator (pymalloc)** manages small Python objects — integers, short strings, list nodes. It maintains its own pool for objects up to 512 bytes.

**Numpy** uses a separate allocator for array data. Large allocations go directly to the OS via `malloc`, which means they *can* be returned to the OS when freed — but whether they actually are depends on the C runtime's own internal pooling behaviour.

**ONNX Runtime** maintains its own memory arena for intermediate tensors during the forward pass. This block stays resident for the lifetime of the session — it is explicitly designed to be reused across inference calls rather than allocated and freed each time.

```python
import onnxruntime as ort
import numpy as np
import resource

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

session = ort.InferenceSession("model.onnx")
print(f"After load:             {get_rss_mb():.0f} MB")

output = session.run(None, {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)})
print(f"After first inference:  {get_rss_mb():.0f} MB")  # Big spike

del output
print(f"After del output:       {get_rss_mb():.0f} MB")  # Barely moves — arena retained

output = session.run(None, {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)})
print(f"After second inference: {get_rss_mb():.0f} MB")  # Barely increases — arena reused
```

Typical output:

```
After load:             480 MB
After first inference:  864 MB   ← +384 MB arena allocation
After del output:       862 MB   ← barely moved
After second inference: 866 MB   ← arena reused, minimal increase
```

That +384 MB that never goes away is not a leak. It is ONNX Runtime's arena, warmed up and waiting for the next inference call.

---

### 2.5 The Distinction That Matters

| Event | Python sees | RSS |
|---|---|---|
| `del x` on a small object | Refcount drops, freed into pymalloc pool | No change |
| `del x` on a large numpy array | Refcount drops, freed via C malloc | May change, may not |
| ONNX arena objects freed | Returned to internal pool | No change |
| Process exits | All arenas returned to OS | Drops to zero |

**The metric you actually want for leak detection is not RSS at a single point in time. It is RSS at the same point across multiple runs.**

---

### 2.6 The Plateau Test: Distinguishing Leak from Arena Retention

A real leak looks like this:

```
Run 1 start:  120 MB
Run 2 start:  145 MB
Run 3 start:  172 MB
Run 4 start:  201 MB  ...keeps climbing
```

Arena retention with correct cleanup looks like this:

```
Run 1 start:  120 MB
Run 2 start:  173 MB  ← one-time init cost (arena warm-up)
Run 3 start:  154 MB
Run 4 start:  154 MB  ...plateau
Run 5 start:  154 MB
```

The plateau is the signal. The one-time jump from run 1 to run 2 is expected — arenas warming up. Once warm, the baseline should stabilise. If it does not, you have a real leak.

```python
import gc
import resource

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

for run in range(10):
    baseline = get_rss_mb()
    print(f"Run {run+1:2d} start: {baseline:.0f} MB")

    run_inference()
    gc.collect()
```

Ten runs is usually enough. Plateau by run 3 or 4 — you are clean. Still climbing at run 10 — you have something to investigate.

---

## Part 3: Frame Objects — The Invisible Anchor

*A function returned. Its local variables didn't.*

---

### 3.1 What Is a Frame Object?

Every time Python calls a function, it creates a **frame object**. You never ask for it. Python creates it automatically to hold everything that function needs while it runs:

- The local variables
- The current instruction — which line is executing
- A reference to the code object (compiled bytecode)
- A reference to the enclosing scope

```python
import sys

def show_my_frame():
    frame = sys._getframe()
    print(type(frame))                   # <class 'frame'>
    print(frame.f_code.co_name)          # show_my_frame
    print(list(frame.f_locals.keys()))   # ['frame']

show_my_frame()
```

When the function returns, Python discards the frame. The local variables lose their references, their refcounts drop, and — in the happy path — they are freed immediately.

**The critical word is *in the happy path*.**

A frame is a Python object like any other. It has a reference count. And if something holds a reference to a frame after the function returns, that frame — and every local variable inside it — stays alive. This is the invisible anchor.

---

### 3.2 How Frames Stay Alive: Three Common Patterns

None of these look dangerous. All of them can anchor large amounts of memory in ML workloads.

#### Pattern 1: Closures

```python
def outer():
    x = [0] * 1_000_000   # large object

    def inner():
        return x[0]        # inner captures x from outer's frame

    return inner

f = outer()
# outer() has returned. But f still references outer's local variable x.
# Therefore outer's frame is still alive. The 1M element list is not freed.
```

When `outer()` returns, the `inner` function holds a reference to `x`, which holds the enclosing scope alive. Python packages this into a **cell object** — a small wrapper that both the outer frame and the inner function share.

The frame, the cell, and the inner function form a cycle:

```
outer frame ──► cell object ──► inner function
     ▲                               │
     └───────────references back─────┘
```

Refcount GC cannot break this. It waits for cyclic GC.

**In ML terms:** replace `x = [0] * 1_000_000` with `output = session.run(...)` returning a 500 MB array. If your inference code defines any nested function that captures a local variable, you have this pattern.

```python
def run_inference(session, input_data):
    output = session.run(None, {"input": input_data})[0]  # 500 MB

    def log_shape():
        print(output.shape)    # captures output — anchors 500 MB

    log_shape()
    return output.argmax()
    # output is NOT freed until cyclic GC runs
```

#### Pattern 2: Context Managers

Context managers — the `with` statement — are everywhere in ML code: logging spans, profiling blocks, timer contexts. They look clean and self-contained. They are also a reliable source of frame cycles.

```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("enter")
    yield
    print("exit")

def do_work():
    large_array = [0] * 1_000_000

    with my_context():
        process(large_array)
    # my_context's generator frame is still alive here
    # large_array is not freed yet
```

A `@contextmanager` is implemented as a generator under the hood. When execution reaches `yield`, the generator's frame is suspended — kept alive so it can resume for the cleanup code. That suspended frame holds a reference back to its calling context. If that context has large local variables, those variables are anchored.

This means that every `with` block using a generator-based context manager creates a temporary frame retention that cyclic GC must clean up. Structlog, OpenTelemetry, and most observability libraries use exactly this pattern for their span blocks.

```python
def run(self):
    with logger.span("inference"):
        input_data = load_data(self.path)      # local var
        output = session.run(None, input_data)  # 500 MB local var
        save_output(output, self.out_path)
    # logger.span()'s generator frame holds a reference chain
    # that reaches input_data and output until GC runs
```

#### Pattern 3: Exceptions and Tracebacks

When Python raises an exception, it builds a traceback object that holds references to the frames of every function in the call stack at that point. Python 3 explicitly deletes the exception variable at the end of an `except` block to handle the most common case — but `sys.exc_info()` and `sys.last_traceback` can still hold references in certain situations.

---

### 3.3 Why This Is Catastrophic in ML

In a web server handling small JSON payloads, frame cycles are largely invisible. A few kilobytes of extra retention, cleaned up by the next GC cycle.

In an ML inference pipeline, the objects anchored by frame cycles are not small. They are input arrays, output arrays, intermediate tensors — hundreds of megabytes each. The frame itself is tiny. The cell objects are tiny. But the local variables those frames *reference* are enormous.

```
generator frame (2 KB)
       │
       └──► run() frame (4 KB)
                  │
                  └──► output array (625 MB)
```

Two tiny objects anchor 625 MB. The cyclic GC has to find and break this chain before any of that memory is available for the next task. If the next task starts first — and it will, unless you trigger GC manually — both tasks' memory is live simultaneously. That is your OOM.

---

### 3.4 Diagnosing Frame Cycles in Your Own Code

**Step 1: Find what's in the cycles**

```python
import gc
from collections import Counter

gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
gc.set_debug(0)

type_counts = Counter(type(obj).__name__ for obj in gc.garbage)
print(type_counts.most_common(10))

gc.garbage.clear()
```

If you see `frame`, `cell`, and `function` in the top types, you have frame cycles. If you see numpy arrays or your own domain objects, the cycle is in application code and should be fixed.

**Step 2: Find which functions those frames belong to**

```python
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
gc.set_debug(0)

frames = [obj for obj in gc.garbage if type(obj).__name__ == 'frame']
for f in frames[:10]:
    print(f"{f.f_code.co_filename}:{f.f_lineno} in {f.f_code.co_name}")
    print(f"  locals: {list(f.f_locals.keys())}")

gc.garbage.clear()
```

If filenames point to your code — fix the cycle. If they point to a third-party library — `gc.collect()` is the appropriate response.

**Step 3: Check cell contents**

```python
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
gc.set_debug(0)

cells = [obj for obj in gc.garbage if type(obj).__name__ == 'cell']
for cell in cells[:10]:
    try:
        content = cell.cell_contents
        print(f"cell holding: {type(content).__name__}")
        if hasattr(content, 'nbytes'):
            print(f"  size: {content.nbytes / 1e6:.1f} MB")
    except ValueError:
        pass

gc.garbage.clear()
```

Cells holding numpy arrays directly — fix the closure. Cells holding `_ClassBuilder` or other library-internal types — `gc.collect()` is appropriate.

---

## Part 4: The Lazy Import Trap

*You can be doing everything right and still have cycles from code you never wrote.*

---

### 4.1 What Is Lazy Loading and Why Libraries Use It

When you write `import nibabel` at the top of your file, Python loads nibabel *and everything nibabel depends on* — every submodule, every optional dependency. For a large scientific library, this can add hundreds of milliseconds to startup time.

The solution most large libraries reach for is **lazy loading** — deferring the import of expensive or optional submodules until they are actually needed.

```python
# Eager loading — happens at import time, always
import heavy_optional_library

# Lazy loading — happens only when first used
def use_heavy_library():
    import heavy_optional_library
    return heavy_optional_library.do_something()
```

Nibabel, for example, uses a utility called `optional_package` — a helper that wraps an optional dependency so it appears available but only actually imports when first accessed. This is a sensible, well-motivated design. The problem is not the *intent* of lazy loading. The problem is what Python's import machinery does under the hood when it finally runs — specifically, when it runs for the first time *inside your inference loop*.

---

### 4.2 What Python's Import Machinery Actually Does

When Python imports a module for the first time, it executes a carefully orchestrated sequence managed by `importlib`. This sequence involves several internal functions, each running in its own frame:

```
_find_and_load()
    └── _load_unlocked()
            └── exec_module()
                    └── _call_with_frames_removed()
                                └── <module>  ← your actual module code
```

Each of these is a real Python function call, which means each creates a real frame object. These frames are alive for the duration of the import. Normally this is fine. The subtlety appears when module-level code creates closures or uses class builders.

During the import, if the module defines classes using a framework like `attrs`, those class definitions create cell objects that reference back to the enclosing import frame. Now you have a cycle:

```
importlib frame (_load_unlocked)
        │
        └──► importlib frame (exec_module)
                      │
                      └──► module-level closure / cell
                                    │
                                    └──► back to exec_module frame
```

This cycle is tiny — the frames and cells hold no large data themselves. Under normal circumstances nobody notices it exists. **But if this lazy import is triggered for the first time during your inference function, the import frames are created while your inference function's frame — and your 500 MB output array — is also alive.**

---

### 4.3 The attrs `_ClassBuilder`: Where the Cycle Forms

`attrs` provides a decorator that transforms a plain class into a fully-featured one with `__init__`, `__repr__`, `__eq__`, and more. It does this by building the class dynamically at class-definition time — which happens when the module is first imported.

The class building process uses a helper called `_ClassBuilder`. The methods it adds are closures. Closures create cell objects. Those cell objects reference the `_ClassBuilder` instance. The `_ClassBuilder` was created inside the `exec_module` frame. The result: every `@attr.define` class in the library creates a small cluster of cell objects in a reference cycle with the import frames.

You can see this directly:

```python
import gc
from collections import Counter

gc.disable()
gc.set_debug(gc.DEBUG_SAVEALL)

def trigger_lazy_import():
    import nibabel.streamlines
    return nibabel.streamlines

trigger_lazy_import()
gc.collect()

types = Counter(type(obj).__name__ for obj in gc.garbage)
print(types.most_common(10))
# You will see: frame, cell, _ClassBuilder, function, dict

frames = [o for o in gc.garbage if type(o).__name__ == 'frame']
for f in frames[:5]:
    print(f"{f.f_code.co_filename}: {f.f_code.co_name}")
# You will see: importlib._bootstrap, optpkg.py, arrayproxy.py

gc.garbage.clear()
gc.set_debug(0)
gc.enable()
```

---

### 4.4 Why This Anchors Your Inference Memory

Here is the full picture of what happens when a lazy import fires inside an inference function:

```
run() called
    │
    ├── input = load_data(path)
    │       └── nibabel lazy import fires for first time
    │               └── importlib frames created
    │                       └── attrs _ClassBuilder runs
    │                               └── cell cycle formed with import frames
    │
    ├── output = session.run(...)    ← 625 MB allocated as local var
    │
    ├── save_output(output, path)
    │
    └── run() returns
            ├── run()'s frame freed — output refcount drops
            │
            └── BUT import frame cycle still alive
                    └── cyclic GC has not run yet
                            └── output array NOT freed
```

The import frames do not directly reference the output array. But they are alive at the same time as the inference frame, and the presence of live import frame cycles in Generation 0 can delay the cyclic GC sweep that would also clean up other pending objects from the same period. The effect is timing-dependent — sometimes the array is freed promptly, sometimes it waits. Under memory pressure, "sometimes" is not good enough.

The direct solution is not to leave it to chance:

```python
import gc

def run(self):
    with logger.span("inference"):
        input_data = load_data(self.path)
        output = session.run(None, input_data)
        save_output(output, self.out_path)

    gc.collect()   # break import frame cycles immediately
```

---

### 4.5 One Important Clarification: This Only Happens Once

The import frames are created the *first time* the lazy import fires. After that, the module is cached in `sys.modules` and subsequent imports return the cached version — no new frames, no new cycles.

This means:
- **Run 1:** lazy import fires → import frames created → cycle exists → `gc.collect()` cleans it up
- **Run 2+:** lazy import hits cache → no new frames → only the smaller context manager cycles remain

This is exactly what shows up in the plateau test. Run 2 behaves slightly differently from run 3 onward, because by run 3 all lazy imports have already fired. The baseline stabilises because no new import cycles are being created — just the same small set of library-internal cycles being cleaned up each time.

---

### 4.6 The Decision Tree for What You Find

```
gc.garbage contains frames?
        │
        ├── Frames from your code?
        │       └── Find the closure or context manager — fix it
        │
        └── Frames from importlib / third-party libraries?
                └── gc.collect() is appropriate
                        │
                        └── Also check: numpy arrays in gc.garbage?
                                ├── Yes → something still holds a direct ref — investigate
                                └── No  → arrays freed once cycles broken — gc.collect() is complete fix
```

---

## Part 5: gc.collect() — Scalpel, Not Sledgehammer

*Calling gc.collect() without understanding why it works is like taking painkillers without treating the injury.*

---

### 5.1 What gc.collect() Actually Does

`gc.collect()` does exactly one thing: it triggers Python's cyclic garbage collector immediately, rather than waiting for Python to decide conditions are right.

```python
import gc

print(gc.get_count())      # (312, 4, 1) — allocations since last collection
print(gc.get_threshold())  # (700, 10, 10) — thresholds that trigger auto-collection

collected = gc.collect()           # all generations — most thorough
collected_gen0 = gc.collect(0)     # youngest only — fastest
```

What it does *not* do:
- It does not free objects with a refcount above zero
- It does not force memory back to the OS
- It does not guarantee RSS drops after the call
- It does not fix reference cycles in your code — it works around them

That last point is the one your colleagues will push back on, and they will be right to. **`gc.collect()` working is evidence that cycles exist. It is not evidence that you have found and fixed them.**

---

### 5.2 The Performance Cost

`gc.collect()` is not free. In a large ML process with a loaded model and thousands of Python objects alive simultaneously, a full collection can take anywhere from a few milliseconds to tens of milliseconds.

```python
import time

start = time.perf_counter()
collected = gc.collect()
cost_ms = (time.perf_counter() - start) * 1000
print(f"gc.collect() freed {collected} objects in {cost_ms:.1f} ms")
```

For an inference pipeline where each task takes several seconds, 10–30 ms is negligible. For a real-time system with tight latency requirements, it is not. Measure before assuming.

```python
# Fine — once per task, task takes seconds
def run_task():
    output = run_inference()
    save_output(output)
    gc.collect()   # cost amortised over several seconds of work

# Not fine — inside tight loop
def run_batch(inputs):
    results = []
    for inp in inputs:
        out = model(inp)
        results.append(out)
        gc.collect()   # called thousands of times — cumulative cost is real
    return results
```

---

### 5.3 When gc.collect() Is the Right Answer

**Case 1: Cycles in third-party library code you do not control**

If your cycle diagnosis shows frames from `importlib`, `nibabel`, `structlog`, `attrs`, or any other library you did not write, you cannot fix the cycle. `gc.collect()` is the right response.

**Case 2: Generator-based context managers from observability libraries**

Structlog, OpenTelemetry, and similar libraries use generator-based context managers for spans. As covered in Part 3, these create frame cycles by design. If your inference pipeline is instrumented — as it should be in production — you will always have some frame cycles to clean up. `gc.collect()` at the end of each task is the standard solution.

**Case 3: Long-running tasks between which memory must be fully reclaimed**

If your process runs sequential tasks that each use a large fraction of available memory, you cannot afford to wait for Python's automatic GC. The window between tasks is the right place to force cleanup.

```python
def run_pipeline(tasks):
    for task in tasks:
        result = process(task)
        save(result)
        del result
        gc.collect()   # next task starts clean
```

---

### 5.4 When gc.collect() Is a Red Flag

**Red flag 1: It frees a large amount and you do not know why**

If `gc.collect()` frees a lot and you have not done a cycle diagnosis, you are flying blind. Run the diagnosis from Part 3 at least once before shipping.

**Red flag 2: Numpy arrays appear in gc.garbage**

If your diagnosis shows numpy arrays directly in `gc.garbage`, the cycle is in your application code. Something is capturing array references in a closure. `gc.collect()` will fix the symptom — but there is a structural issue worth finding.

**Red flag 3: Your own domain objects appear in gc.garbage**

If you see your own classes in `gc.garbage`, you have a circular reference in your object model. This is fixable: find the cycle, use `weakref` for the back-reference, or restructure to eliminate the mutual dependency.

**Red flag 4: Memory still grows after many runs**

If memory grows across dozens of runs and `gc.collect()` does not bring it back down, you have a true reference leak — something accumulating in a list, cache, global, or registry that never gets cleared. `gc.collect()` cannot help here. You need `tracemalloc`.

```python
import tracemalloc

tracemalloc.start()
# run your pipeline for N iterations
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)
```

---

### 5.5 The Pre-Ship Checklist

Before shipping any code that calls `gc.collect()` in production, work through this checklist.

**☐ Step 1: Measure what it frees**

```python
gc.collect()   # warm run
run_inference_task()
collected = gc.collect()
print(f"Freed {collected} objects")
```

**☐ Step 2: Run the cycle diagnosis**

Check `gc.garbage` after `gc.DEBUG_SAVEALL`. Answer: are there numpy arrays? Your own objects? Where do the frames come from?

**☐ Step 3: Run the plateau test**

Ten consecutive tasks, log `start_mb` at the beginning of each. Confirm baseline plateaus within the first few runs.

**☐ Step 4: Measure the cost**

```python
start = time.perf_counter()
gc.collect()
print(f"Cost: {(time.perf_counter() - start) * 1000:.1f} ms")
```

**☐ Step 5: Document your reasoning**

This is the step most engineers skip and the one that saves the most time six months later.

```python
# At end of run() in inference pipeline.
#
# Necessary because the NIfTI loading library's optional_package() lazy
# import mechanism creates reference cycles between importlib frames and
# attrs _ClassBuilder cell objects when submodules are first loaded.
# These cycles are in third-party library code and cannot be fixed at
# the application level.
#
# Without this call, the cyclic GC may not run before the next inference
# task begins, causing peak memory from consecutive tasks to overlap and
# exceed container limits (OOMKilled, exit code 137).
#
# Diagnosis: 0 numpy arrays in gc.garbage, 74 frame objects all from
# importlib/_bootstrap and library internals, 194 cell objects holding
# _ClassBuilder and dict types. 10-run plateau test confirms no
# accumulation across tasks.
gc.collect()
```

---

## Part 6: The Investigation — Putting It All Together

*Back to the pipeline that kept dying.*

---

### The Setup

The pipeline runs inference on large medical images. It is containerised — each stage runs in its own Docker container with a fixed memory limit. The workflow runs multiple stages in sequence, each loading an ONNX model, running inference on a NIfTI format medical image, and writing output for the next stage to consume.

The symptom was consistent and reproducible. Run the workflow once — everything completes. Run it a second time while the first is still processing — one of the inference containers gets OOM-killed. Exit code 137. `OOMKilled: true` in the container status.

The immediate suspicion was the session cache in our model loader. My colleague was convinced it was holding references to arrays and preventing garbage collection. It was a reasonable hypothesis. It turned out to be wrong.

---

### Step 1: Instrument First, Guess Later

We added RSS tracking at seven checkpoints using `/proc/self/status`:

```python
def get_rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024
    return 0

# M1: start of run()
# M2: after loading input data
# M3: after every 5th batch during inference
# M4: after np.concatenate() — full output assembled
# M5: after saving output
# M6: after session cache clear
# M7: after del + gc.collect()
```

The first full run:

| Marker | RSS (MB) | Delta | Notes |
|--------|----------|-------|-------|
| M1 Start | 120 | — | Clean baseline |
| M2 After load | 252 | +132 | input data ≈ 125 MB |
| M3 Batch 5 | 792 | +540 | +384 MB unexpected |
| M3 Batch 20 | 1364 | — | outputs accumulating |
| M4 After concat | 1989 | +625 | output ≈ 625 MB |
| M5 After save | 1989 | 0 | no change |
| M6 After cache clear | 1975 | **-14** | cache = only 14 MB |
| M7 After gc.collect | 1350 | **-639** | output array freed |

Two things were immediately clear. The session cache freed only 14 MB — the hypothesis was wrong. And `gc.collect()` freed 639 MB — something was in a cycle, and it was large.

---

### Step 2: Rule Out the Obvious Suspects

**Suspect 1: numpy arrays held by external referrers**

```python
referrers = gc.get_referrers(raw_output)
print(f"Referrers: {len(referrers)}")   # 0 external referrers
```

Zero external referrers. Nothing in application code was holding the array. The reference was inside a cycle, invisible to `get_referrers`.

**Suspect 2: sys.last_traceback**

```python
print(sys.last_traceback)   # None
```

Ruled out.

**Suspect 3: Logger holding references**

```python
referrer_types = [type(r).__name__ for r in gc.get_referrers(raw_output)]
print(referrer_types)   # [] — logger not involved
```

All three suspects ruled out in under an hour.

---

### Step 3: The 10-Run Accumulation Test

To answer the core challenge — is `gc.collect()` masking a slow leak? — we ran ten consecutive tasks and logged `start_mb`:

| Run | Start (MB) | Peak (MB) | End (MB) |
|-----|-----------|-----------|----------|
| 1 | 120 | 1989 | 1350 |
| 2 | 173 | 1999 | 1373 |
| 3 | 154 | 1999 | 1373 |
| 4–10 | 154 | 1999 | 1373 |

The baseline plateaued at 154 MB from run 3 onwards. Peak was perfectly stable. The +53 MB jump from run 1 to 2 was the ONNX Runtime arena warming up — expected and bounded.

This ruled out a slow leak with high confidence. The question shifted: *what exactly is gc.collect() freeing?*

---

### Step 4: Inspect gc.garbage

```python
gc.set_debug(gc.DEBUG_SAVEALL)
collected = gc.collect()
gc.set_debug(0)

types = Counter(type(obj).__name__ for obj in gc.garbage)
print(f"Collected: {collected} objects")
print(f"Types: {types.most_common(10)}")

arrays = [o for o in gc.garbage if type(o).__name__ == 'ndarray']
print(f"Numpy arrays in cycles: {len(arrays)}")

gc.garbage.clear()
```

Output:

```
Collected: 639 objects
Types: [('function', 24329), ('dict', 14865), ('tuple', 14325),
        ('cell', 4636), ('ReferenceType', 4482), ('frame', 74), ...]
Numpy arrays in cycles: 0
OrtValue objects in cycles: 0
```

**Zero numpy arrays in gc.garbage.** The 625 MB array was not directly in a cycle — it was freed as a *consequence* of breaking cycles. Once the frames and cells were freed, their reference chains collapsed, and the array's refcount finally hit zero.

**74 frame objects and 4636 cell objects.** These were the cycles. Small Python objects anchoring a very large allocation.

---

### Step 5: Find the Frame Sources

```python
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
gc.set_debug(0)

frames = [o for o in gc.garbage if type(o).__name__ == 'frame']
func_counts = Counter(f.f_code.co_name for f in frames)
print(func_counts.most_common(10))

for f in frames[:5]:
    print(f"{f.f_code.co_filename}: {f.f_code.co_name}")

gc.garbage.clear()
```

Output:

```
Top functions in stuck frames:
_call_with_frames_removed  (4x)  — importlib/_bootstrap.py
<module>                   (3x)  — module loading
exec_module                (3x)  — importlib/_bootstrap_external.py
_load_unlocked             (3x)  — importlib/_bootstrap.py
optional_package           (2x)  — nibabel/optpkg.py
_find_and_load             (2x)  — importlib/_bootstrap.py

Files: optpkg.py, _compression.py, arrayproxy.py, analyze.py
```

Every single stuck frame was from Python's import machinery or the NIfTI library's internals. Not one frame from our inference code. Not one from any observability library.

---

### Step 6: Inspect the Cell Contents

```python
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
gc.set_debug(0)

cells = [o for o in gc.garbage if type(o).__name__ == 'cell']
content_types = []
for cell in cells[:20]:
    try:
        content_types.append(type(cell.cell_contents).__name__)
    except ValueError:
        content_types.append("empty")

print(Counter(content_types).most_common())
gc.garbage.clear()
```

Output:

```
[('_ClassBuilder', 4), ('dict', 3), ('function', 2), ('MagicProxy', 1)]
```

`_ClassBuilder` — the library's use of `attrs` for dynamic class construction at import time. `MagicProxy` — a placeholder object for optional dependencies. Both are library internals. The picture was complete.

---

### The Complete Picture

During the first inference task, the NIfTI loading library's lazy import system fired for the first time. Python's `importlib` machinery ran through its standard sequence, creating frame objects at each level. Inside the module code, `attrs`-powered class definitions ran, creating `_ClassBuilder` instances and cell objects that formed cycles with the import frames.

These cycles are small — a few kilobytes total. Normally they would be cleaned up quickly by automatic cyclic GC. But automatic GC runs on an allocation-count threshold, not a schedule. In the window between the end of one inference task and the beginning of the next, the threshold had not been reached. The import frame cycles were still alive.

The 625 MB output array had its refcount drop to zero when `run()` returned. But because the cyclic GC had not run yet, the memory was not freed before the next task began allocating its own working memory. Two tasks' peaks overlapped. Container limit exceeded. OOM.

`gc.collect()` at the end of `run()` forces the cyclic GC to run immediately. The import frame cycles are broken. The reference chains collapse. The array's memory is freed before the next task begins. The peaks no longer overlap.

---

### What This Investigation Ruled Out — and Confirmed

| Hypothesis | Result | Evidence |
|---|---|---|
| Session cache holding arrays | Ruled out | Cache cleared 14 MB only |
| Numpy arrays directly in cycles | Ruled out | 0 arrays in gc.garbage |
| OrtValue / ONNX Runtime cycles | Ruled out | 0 OrtValue in gc.garbage |
| Logger holding references | Ruled out | 0 referrers on arrays |
| sys.last_traceback | Ruled out | None |
| Slow accumulating leak | Ruled out | start_mb plateaus at run 3 |
| Application-level frame cycles | Ruled out | 0 frames from application code |
| NIfTI library lazy import cycles | **Confirmed** | All 74 frames from importlib + library internals |
| attrs _ClassBuilder cell cycles | **Confirmed** | Cell contents: _ClassBuilder, MagicProxy |
| gc.collect() appropriate fix | **Confirmed** | Library-internal cycles, unfixable from application code |

---

### Why This Pattern Will Find You Again

The specific libraries here are not the point. The pattern is.

Any Python ML pipeline that combines:

1. A scientific or data library with lazy-loaded optional dependencies
2. Classes built with `attrs`, `pydantic`, or similar dynamic class frameworks
3. Large array allocations in the same execution context as first-time imports
4. Sequential tasks running in the same long-lived process

...is susceptible to exactly this. The libraries will differ. The frame sources will differ. But the mechanism — import frames cycling with class-builder closures, anchoring memory past the end of the task that triggered the import — will be the same.

The investigation methodology is what transfers: instrument before guessing, inspect `gc.garbage` before assuming the cause, trace frame sources before shipping `gc.collect()`, run the plateau test before calling it fixed.

---

## Part 7: The Toolkit — What to Reach for and When

*A practical reference for when you are in the middle of an investigation.*

---

### Quick Reference

| Tool | Question it answers | When to use |
|---|---|---|
| `/proc/self/status` | Where does memory grow? | Always first |
| Plateau test | Leak or arena? Is gc working? | Before investigating internals |
| `gc.get_referrers()` | What holds this specific object? | When you have a suspect object |
| `gc.DEBUG_SAVEALL` + `gc.garbage` | What types are in cycles? | After confirming gc.collect() frees a lot |
| Frame inspection | Whose frames are stuck? | After finding frames in gc.garbage |
| Cell inspection | What are closures capturing? | After finding cells in gc.garbage |
| `tracemalloc` | Where do growing allocations come from? | When plateau test shows ongoing growth |
| `objgraph` | What is the full reference chain? | When other tools give incomplete picture |

---

### `/proc/self/status` — RSS Tracking

```python
def get_rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024
    return 0

# macOS alternative
import resource
def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
```

**Limitation:** Does not distinguish freed-but-retained (arena) from genuinely leaked memory. Use it to find *where* memory grows, then use other tools to find *why*.

---

### The Plateau Test

```python
import gc, resource

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

for run in range(10):
    start = get_rss_mb()
    run_your_inference_task()
    gc.collect()
    print(f"Run {run+1:2d} | start: {start:.0f} MB")

# Plateau by run 3–4 → healthy
# Still climbing at run 10 → real leak, investigate further
```

---

### gc.garbage + gc.DEBUG_SAVEALL

```python
import gc
from collections import Counter

gc.set_debug(gc.DEBUG_SAVEALL)
collected = gc.collect()
gc.set_debug(0)

types = Counter(type(obj).__name__ for obj in gc.garbage)
for type_name, count in types.most_common(10):
    print(f"  {type_name}: {count}")

arrays = [o for o in gc.garbage if type(o).__name__ == 'ndarray']
print(f"Numpy arrays in cycles: {len(arrays)}")

gc.garbage.clear()   # always clear after inspection
```

**Always call `gc.garbage.clear()` after inspection.** If you leave objects in `gc.garbage`, they accumulate and you will see stale results in subsequent calls.

**Reading the output:**

| Types in gc.garbage | Interpretation | Action |
|---|---|---|
| `ndarray`, your domain objects | Application-level cycle | Fix the closure or circular reference |
| Frames from your files | Your code creates cycles | Restructure |
| Frames from importlib / library files | Third-party import cycle | gc.collect() appropriate |
| `_ClassBuilder`, `MagicProxy` | Library class-building cycle | gc.collect() appropriate |

---

### Frame Inspection

```python
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
gc.set_debug(0)

frames = [o for o in gc.garbage if type(o).__name__ == 'frame']
func_counts = Counter(f.f_code.co_name for f in frames)
print(func_counts.most_common())

for f in frames[:5]:
    print(f"  {f.f_code.co_filename}: {f.f_code.co_name}:{f.f_lineno}")
    print(f"  locals: {list(f.f_locals.keys())}")

gc.garbage.clear()
```

---

### Cell Inspection

```python
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
gc.set_debug(0)

cells = [o for o in gc.garbage if type(o).__name__ == 'cell']
content_types = []
large_contents = []

for cell in cells:
    try:
        content = cell.cell_contents
        content_types.append(type(content).__name__)
        if hasattr(content, 'nbytes') and content.nbytes > 10 * 1024 * 1024:
            large_contents.append((type(content).__name__, content.nbytes / 1e6))
    except ValueError:
        content_types.append("empty")

print(Counter(content_types).most_common(10))

if large_contents:
    print("Large objects in closures (RED FLAG):")
    for type_name, mb in large_contents:
        print(f"  {type_name}: {mb:.1f} MB")

gc.garbage.clear()
```

---

### tracemalloc — For True Leaks

```python
import tracemalloc

tracemalloc.start()

snapshot1 = tracemalloc.take_snapshot()
run_inference_task()
snapshot2 = tracemalloc.take_snapshot()

top_stats = snapshot2.compare_to(snapshot1, 'lineno')
print("Growth between runs:")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

**When to use:** only when the plateau test shows memory still growing after ten runs and `gc.collect()` is not bringing it back down. `tracemalloc` only tracks Python-level allocations — it cannot see into C extension memory.

---

### objgraph — For Complex Reference Chains

```python
import objgraph

# Show what holds your array alive
objgraph.show_backrefs(output_array, max_depth=5, filename='refs.png')

# Show what has grown since last check
objgraph.show_growth(limit=10)
```

**When to use:** when `gc.get_referrers()` gives a confusing picture and you need to see the full reference chain visualised. Requires `pip install objgraph` and optionally `graphviz`.

---

### The Investigation Sequence

```
Memory problem suspected
        │
        ▼
RSS tracking — where in the pipeline does memory grow?
        │
        ▼
Plateau test — leak or arena? Is gc.collect() reclaiming?
        │
        ├── Plateaus → gc.collect() working
        │       └── gc.DEBUG_SAVEALL → inspect gc.garbage
        │               ├── Library frames only → ship gc.collect()
        │               └── Your objects → investigate ↓
        │
        └── Still growing → gc.collect() not enough
                └── tracemalloc → find the accumulating allocation

Found frames in gc.garbage?
        └── Frame inspection → whose code?
                ├── Yours → fix the cycle
                └── Library → gc.collect() appropriate

Found cells in gc.garbage?
        └── Cell inspection → what are closures holding?
                ├── Large arrays → fix the closure
                └── Library internals → gc.collect() appropriate

Picture still unclear?
        └── objgraph → full reference chain visualisation
```

---

## Part 8: Closing

*Don't fear the garbage collector. Don't love it too much. Understand it and use it wisely.*

---

Python's memory model is not broken. Every behaviour covered in this article — reference counting, cyclic GC, memory arenas, frame cycles, lazy import patterns — exists for a reason. Refcounting is fast and deterministic. Arenas avoid expensive round-trips to the OS. Lazy loading keeps startup times reasonable. Generator-based context managers make resource cleanup composable and readable. The `attrs` class builder makes defining complex data classes concise and correct. None of these are mistakes. They are deliberate tradeoffs made by smart people solving real problems.

What catches engineers off guard is not that these tradeoffs exist — it is that they interact in ways that only become visible at scale. A frame cycle from a context manager is invisible in a script that runs once. It becomes a production incident in a pipeline that runs the same function a thousand times a day on inputs measured in gigabytes. The mechanism is the same. The consequences are not.

This is why understanding the model matters more than memorising the symptoms. Symptoms change — different libraries, different Python versions, different workload shapes produce different surface behaviour. The model stays the same. If you understand why refcounting cannot break cycles, you will recognise the pattern whether the cycle comes from nibabel, from your own logging library, or from something nobody has written about yet. If you understand what arenas are, you will not spend two days investigating a "leak" that is expected allocator behaviour. If you understand what a frame object is and why it outlives the function that created it, you will know where to look when `del output` does not free what you expected.

The investigation methodology in Part 6 is reusable precisely because it is grounded in the model. Instrument first. Check referrers. Inspect `gc.garbage`. Trace frame sources. Run the plateau test. Each step asks a question that the model tells you is worth asking, and each answer either closes a hypothesis or points to the next question. That sequence works regardless of which specific library created the cycle, which version of Python you are running, or which ML framework you are using.

And `gc.collect()` — the tool that started the whole conversation — turns out to be neither the villain nor the hero. It is a scalpel. Used without understanding, it papers over problems that will resurface in harder-to-diagnose forms. Used with the full picture — after running the diagnosis, after tracing the frame sources, after confirming the cycles are in library code you do not control — it is the correct, complete, and appropriate solution. The difference between those two uses is not the line of code. It is everything that comes before it.

The engineers who build reliable ML systems in Python are not the ones who avoid memory management. They are the ones who understand it well enough to know when to worry and when not to — and who leave a comment in the code explaining which one applies and why.

---

*If this article saved you a debugging session, the best thing you can do is share it with the engineer on your team who will hit this next. They are out there. This will happen to them.*