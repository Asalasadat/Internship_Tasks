# 📚 Week 1 Summary — Python Foundations for Data Science

## Environment & Tools

Every AI and Data Science project starts with a reproducible Python environment. Using tools like **venv** or **conda**, we isolate project dependencies and avoid conflicts between different projects. The used library versions are documented in a `requirements.txt` file, allowing anyone to recreate the same environment and achieve consistent results.

**Jupyter Notebook** is the main workspace for data science workflows. It combines executable code cells with Markdown cells for documentation, creating a structured report rather than an unorganized collection of code. **Git and GitHub** provide version control, collaboration, and a professional portfolio for tracking and sharing projects.

---

## Python Fundamentals

Python fundamentals include working with core data types such as numbers, strings, lists, tuples, dictionaries, and sets. Program execution is controlled using conditional statements (`if`, `elif`, `else`) and loops (`for`, `while`).

Functions allow us to organize reusable logic with clear inputs and outputs. **List comprehension** provides a concise way to create lists using a single line of code instead of traditional multi-line loops.

Object-Oriented Programming (OOP) concepts such as **classes, attributes, and methods** are essential because many machine learning libraries, including **Scikit-learn** and **PyTorch**, are built around these principles.

---

## NumPy — Numerical Computing

**NumPy** is the foundation for many data science and machine learning libraries, including Pandas, Scikit-learn, TensorFlow, and PyTorch.

The core object in NumPy is the **ndarray**, a fast and memory-efficient multi-dimensional array compared to standard Python lists. Arrays can be created using different methods such as `array()`, `zeros()`, and `arange()`.

NumPy supports powerful operations including:

- **Indexing and slicing** for accessing array elements.
- **Vectorization**, which allows performing operations on entire arrays without writing explicit loops, improving both speed and readability.
- **Broadcasting**, which enables operations between arrays with compatible but different shapes.

---

## Pandas — Tabular Data Analysis

**Pandas** is the primary tool for working with structured and tabular data. It provides two main data structures:

- **Series:** A labeled one-dimensional data structure (single column).
- **DataFrame:** A two-dimensional table containing rows and columns, similar to data stored in CSV files, Excel sheets, or databases.

The typical workflow begins with loading and exploring data using functions such as:

- `read_csv()`
- `head()`
- `info()`
- `describe()`

Data preprocessing includes selecting and filtering rows and columns, handling missing values, and removing duplicates using:

- `fillna()`
- `dropna()`
- `drop_duplicates()`

The `groupby()` function enables the **split-apply-combine** process, allowing data to be grouped and analyzed to discover patterns and trends.

---

## Matplotlib — Data Visualization

Data tables can hide important patterns that become clear through visualization. **Matplotlib** is a fundamental library for creating meaningful plots.

The four main visualization types are:

- **Line Plot:** Used to track changes over a continuous axis.
- **Scatter Plot:** Used to analyze relationships between two variables.
- **Bar Chart:** Used to compare values across categories.
- **Histogram:** Used to understand the distribution of a single variable.

The `subplots()` function allows multiple visualizations to be displayed together for easier comparison.

A key principle in data visualization:

> A plot without labels, titles, and clear explanations is not a result — it is just a picture.
