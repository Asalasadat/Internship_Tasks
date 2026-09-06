# Week 7 — CNNs, RNNs & Transformers — Sprint 2

**BinX Tech · AI & Machine Learning Internship Program**
*Convolutional, Recurrent & Attention-Based Architectures*

The second sprint of the Phase 3 capstone: the specialized deep-learning architectures. Interns learn CNNs for images, RNNs/LSTMs for sequences, and the attention mechanism behind Transformers — then advance their project's core model and run the full Sprint 2 review cycle.

`PHASE 3 — SPRINT 2` · `40 HOURS` · `5 TRAINING DAYS` · `HANDS-ON`

---

## 📋 Week 7 Overview

| | |
|---|---|
| **Week** | Week 7 of 10 — Phase 3: Deep Learning & Applied Project, Sprint 2 |
| **Total Hours** | 40 hours (full-time track) / 20 hours (part-time track, Weeks 13–14 combined) |
| **Format** | On-site / Remote / Hybrid — notebooks in Jupyter/Colab, committed to GitHub via feature branches |
| **Focus** | CNNs for images, RNNs/LSTMs for sequences, the attention mechanism and Transformers, pre-trained models; Sprint 2 core model development of the Phase 3 capstone project |
| **Prerequisite** | Weeks 1–6 completed; Sprint 1 reviewed and retrospective completed; baseline + first neural network in place |
| **Mentor Supervision** | Mentor Code & Notebook Review mid-sprint (Day 3) via pull request; Sprint Review + Retrospective on Day 5 |

## 🔁 Sprint 2 Structure

Week 7 is Sprint 2 of the Phase 3 capstone project, following the program's standard sprint cycle:

| Event | Description |
|---|---|
| **Sprint Planning (Day 1)** | Define the Sprint 2 goal and backlog (core model development), incorporating the Sprint 1 retrospective improvement. |
| **Daily Stand-up (Daily)** | 3-minute update: what was completed, what is next, any blockers. |
| **Mentor Code & Notebook Review (Day 3)** | Mentor reviews the notebook and code via GitHub pull request; structured comments on architecture and results. |
| **Sprint Review (Day 5)** | Demo the improved core model and its metrics. Incomplete tasks move to Sprint 3 with documented reasons. |
| **Sprint Retrospective (Day 5)** | What went well, what to improve, and one specific action for Sprint 3. |

## 🎯 Week 7 Learning Objectives

- Explain why fully connected networks fail on images and how convolution solves it.
- Build a CNN with convolution, pooling, and dense layers, and apply transfer learning.
- Explain how RNNs and LSTMs process sequential data and why LSTMs handle long sequences better.
- Explain the attention mechanism and the core idea behind the Transformer architecture.
- Advance the Phase 3 project's core model using the architecture appropriate to its data type.
- Run the full Sprint 2 cycle: planning, stand-ups, mentor review, Sprint Review, and Retrospective.

## 🗓️ Daily Schedule

| Day | Hours | Topic Focus |
|---|---:|---|
| **Day 1** | 8 hrs | Sprint 2 planning; why CNNs — convolution, filters, feature maps |
| **Day 2** | 8 hrs | Building a CNN: pooling, architecture; transfer learning with pre-trained models |
| **Day 3** | 8 hrs | Sequential data: RNNs and LSTMs; Mentor Code & Notebook Review |
| **Day 4** | 8 hrs | The attention mechanism and the Transformer architecture; pre-trained transformers |
| **Day 5** | 8 hrs | Advancing the project's core model; Sprint Review & Retrospective |

---

## 📚 Day-by-Day Curriculum

### Day 1 — Sprint 2 Planning & Convolutional Neural Networks (8 hours)

**Learning Objectives**
- Complete Sprint 2 planning and define the core-model backlog.
- Explain why dense networks fail on images.
- Explain convolution, filters, and feature maps, and why CNNs are efficient.

**Key Topics:** Sprint 2 planning (goal, backlog, carrying forward the Sprint 1 retrospective) · Why dense networks fail on images · Convolution (filters, feature maps, stride, padding) · Parameter sharing and translation invariance · The feature hierarchy CNNs learn

A 200×200 color image has 120,000 numbers — feeding it into a dense network needs millions of first-layer weights and throws away spatial structure. Convolution slides a small filter (e.g. 3×3) across the image, computing a dot product at each position — the Week 2 dot product, applied locally and repeatedly — producing a feature map of where a pattern appears. Convolution wins on **parameter sharing** (one filter reused everywhere) and **translation invariance** (a pattern is detected no matter where it appears). CNNs learn a hierarchy: early layers detect edges, middle layers combine them into shapes/textures, deep layers recognize whole objects.

**Hands-On Lab:** Complete Sprint 2 planning and select backlog tasks → apply a hand-defined edge-detection filter to a sample image and visualize the feature map → explain in Markdown why a shared filter needs far fewer weights than a dense layer → confirm whether the project's data type calls for a CNN, RNN/Transformer, or the Week 6 dense network.

**Tools:** TensorFlow/Keras · NumPy · Matplotlib · Jupyter/Colab · Git & GitHub

---

### Day 2 — Building CNNs & Transfer Learning (8 hours)

**Learning Objectives**
- Build a full CNN with convolution, pooling, and dense layers.
- Apply data augmentation to reduce overfitting on image data.
- Use transfer learning with a pre-trained model to get strong results from little data.

**Key Topics:** Pooling (shrinking feature maps) · Full CNN architecture (conv + pool + flatten + dense) · Data augmentation for computer vision · Transfer learning with ResNet / EfficientNet / MobileNet · Freezing and fine-tuning pre-trained layers

Pooling (e.g. max pooling over 2×2 windows) shrinks feature maps, reduces computation, and adds robustness to small shifts. A typical CNN alternates conv+pool blocks, then flattens into dense layers for classification. Data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`) artificially expands small image datasets — the standard first defense against overfitting. Transfer learning reuses a model already trained on millions of images (e.g. `MobileNetV2` with `weights="imagenet"`, `base.trainable = False`), keeping its learned features and training only a new classification head — the most practical technique in applied computer vision.

**Hands-On Lab:** Build and train a small CNN from scratch on an image dataset and record its accuracy → add data augmentation and compare validation curves → apply transfer learning with a frozen pre-trained model (e.g. MobileNetV2) and compare accuracy and training time → document which approach performed best and why.

**Tools:** TensorFlow/Keras · Pre-trained models (Keras Applications) · Matplotlib · Jupyter/Colab (GPU)

---

### Day 3 — RNNs & LSTMs for Sequential Data (8 hours)

**Learning Objectives**
- Explain why sequential data needs an order-aware architecture.
- Explain how an RNN's hidden state carries memory across a sequence.
- Explain the vanishing-gradient problem and how LSTMs solve it.

**Key Topics:** Why order matters in sequential data · RNNs (processing sequences with a hidden state) · The vanishing gradient problem · LSTMs and GRUs (gated memory) · Embeddings for representing text

Sequential data (text, time series, audio) has a property CNNs don't handle: order matters, and each element depends on what came before ("the movie was not good" ≠ "good, the movie was not"). An RNN processes a sequence one element at a time, maintaining a hidden state passed forward — but plain RNNs struggle with long sequences because gradients shrink exponentially across time steps (vanishing gradient). An **LSTM** adds a gated memory cell — gates learn what to keep, forget, and output — solving this in practice; a **GRU** is a simpler, faster variant with similar performance. If the capstone project is text-based (e.g. Sentiment Analysis), an LSTM (or Day 4's transformer) is the natural core model.

**Hands-On Lab:** Build and train an LSTM on a sequential dataset (text sentiment or a time series) and record its metric → compare against a plain RNN or non-sequential baseline → explain in Markdown why order-awareness helped → open a pull request for the mid-sprint Mentor Code & Notebook Review.

**Tools:** TensorFlow/Keras (LSTM) · Matplotlib · Git & GitHub (pull request)

---

### Day 4 — Attention & Transformers (8 hours)

**Learning Objectives**
- Explain the limitation of RNNs that Transformers overcome.
- Explain the attention mechanism and why it is powerful and parallelizable.
- Use a pre-trained Transformer from Hugging Face for a text task.

**Key Topics:** The limitation of step-by-step RNNs · The attention mechanism (self-attention over all positions) · The Transformer architecture and positional encoding · Pre-trained transformers (BERT, DistilBERT, GPT-2) · Hugging Face Transformers for fine-tuning and inference

Even LSTMs process sequences step by step — slow, with no parallelism, and still imperfect at very long-range dependencies. The Transformer (2017) removed recurrence entirely: **attention** lets every element look directly at every other element and weigh its relevance (self-attention), all positions are processed at once (parallelism), and there's direct access to distant elements with no vanishing gradient (long-range context). A Transformer stacks attention + feed-forward layers plus positional encoding. Just as Day 2 reused pre-trained CNNs, Hugging Face's `pipeline()` makes loading pre-trained Transformers (BERT, DistilBERT, GPT-2) for text tasks straightforward.

**Hands-On Lab:** Load a pre-trained transformer with the Hugging Face pipeline and run it on sample text → if the project is text-based, apply the transformer to the project data and compare to the Day 3 LSTM → explain in Markdown how attention differs from an RNN's step-by-step memory → record which architecture (LSTM vs. transformer) will serve as the project's core model.

**Tools:** Hugging Face Transformers · TensorFlow / PyTorch · Jupyter/Colab (GPU)

---

### Day 5 — Advancing the Core Model & Sprint Review (8 hours)

**Learning Objectives**
- Select the architecture that fits the project's data type.
- Advance and tune the project's core model to beat the baseline, logging experiments.
- Complete the full Sprint 2 review and retrospective cycle.

**Key Topics:** Matching architecture to data type · Advancing the core model over Sprint 1 · Logging experiments and comparing to the baseline · Assembling Sprint Review evidence · Sprint Review and Retrospective

**Choosing the Right Architecture:**

| Project Data Type | Core Architecture |
|---|---|
| Tabular (churn, house price, fraud) | Dense network (Week 6) or gradient boosting; deep learning optional |
| Images (image classifier) | CNN with transfer learning (Day 2) |
| Text (sentiment analysis) | LSTM or a pre-trained transformer (Days 3–4) |
| Sequences / time series | LSTM/GRU (Day 3) |

Not every project needs a Transformer or CNN — matching the architecture to the data, rather than reaching for the most complex option, is the mark of real understanding. Sprint 2's deliverable is a meaningfully improved core model over Sprint 1, with every experiment's configuration and metric logged (MLflow or notebook) and compared to the baseline.

**Hands-On Lab:** Confirm and justify the core architecture for the project → train and tune the improved core model, logging each experiment → assemble a metric table comparing the Sprint 2 model to the Week 6 baseline and the Sprint 1 network → ensure all work is committed and the PR merged after mentor approval → present the Sprint Review, then write the Retrospective with one concrete change for Sprint 3.

**Tools:** TensorFlow/Keras or Hugging Face · MLflow (experiment logging) · Matplotlib · Git & GitHub

---

## 📦 Week 7 Deliverables

By the end of Week 7 (Sprint 2), every intern must submit the following to their mentor and GitHub repository:

- [ ] The Sprint 2 plan and a convolution demo showing a filter producing a feature map.
- [ ] A CNN notebook comparing a from-scratch CNN, augmentation, and transfer learning on image data.
- [ ] An LSTM notebook on sequential data compared against a non-sequential baseline.
- [ ] A pre-trained transformer applied to a text task via Hugging Face, with a short attention-vs-RNN explanation.
- [ ] The advanced project core model with a metric table beating the Week 6 baseline, plus logged experiments.
- [ ] All Sprint 2 work committed to GitHub via reviewed pull requests, plus the written Sprint Retrospective.

## 📊 Week 7 Evaluation Criteria

Scored by the assigned mentor at the Sprint 2 review using the criteria below (extracted from the program's 100-point internal rubric).

| Criterion | 50–69 Developing | 70–84 Proficient | 85–100 Excellent |
|---|---|---|---|
| Understanding of architectures | Names architectures, unclear on when to use each | Explains CNNs, RNNs/LSTMs, and attention clearly | Deep understanding; matches architecture to data with justification |
| Deep learning proficiency | Builds models with heavy guidance | Builds CNNs/LSTMs and uses pre-trained models independently | Fluent; applies transfer learning and tuning effectively |
| Core model quality vs. baseline | Model built but weak baseline comparison | Improved model clearly documented vs. baseline | Rigorous; tuned core model decisively beats baseline |
| Sprint discipline | Misses stand-ups or planning steps | Completes full sprint cycle: plan, stand-ups, review, retro | Proactive sprint management; flags blockers early |
| Notebook & Git workflow | Sporadic commits, weak narrative | Regular commits, PR review, clear Markdown | Consistent, descriptive, reproducible, well-organized |
| Attendance & punctuality | 3–6 absences | 1–2 absences, on time | Perfect attendance, proactive |

## 🧰 Technical Stack — Week 7

| Area | Tools |
|---|---|
| Computer Vision | TensorFlow/Keras (`Conv2D`, `MaxPooling2D`), Keras Applications (ResNet, EfficientNet, MobileNet) |
| Sequence Models | TensorFlow/Keras (`LSTM`, `GRU`, `Embedding`) |
| Transformers | Hugging Face Transformers (BERT, DistilBERT, GPT-2) |
| Experiment Tracking | MLflow |
| Compute & Workflow | Google Colab (GPU), Matplotlib, Git & GitHub (feature branches + pull requests) |

## 📁 Repository Structure

```text
.
├── Day1-Sprint2Planning-Convolution.ipynb
├── Day2-CNN-TransferLearning.ipynb
├── Day3-RNN-LSTM.ipynb
├── Day4-Attention-Transformers.ipynb
├── Day5-CoreModel-SprintReview.ipynb
├── requirements.txt
└── README.md
```

## 📝 Notes & Best Practices

Each architecture this week is the right tool for one data shape: CNNs exploit spatial structure, RNNs/LSTMs exploit order, Transformers use attention for direct long-range context. The professional skill is matching the architecture to the data — not reaching for the most complex model. Whatever the choice, it only counts if it beats the Week 6 baseline, logged and documented.

---

*Prepared by BinX Tech — Palestine | Nablus*
