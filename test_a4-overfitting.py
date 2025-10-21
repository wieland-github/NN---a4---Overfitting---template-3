# test_overfitting_todos.py
# -*- coding: utf-8 -*-
"""
PyTests für a4_A_Overfitting-2.ipynb, fokussiert NUR auf die TODOs:
- generate_sine_data
- PolynomialGradientDescentModel
Es werden ausschließlich die Definitionszellen geladen, die diese Symbole enthalten.
Demos/Plots/Training-Zellen werden übersprungen.
"""

import io
import json
import numpy as np
import pytest

NB_PATH = "a4_A_Overfitting-2.ipynb"

# ---------------------------
# Loader: nur relevante Defs
# ---------------------------

INCLUDE_PATTERNS = [
    "def generate_sine_data",
    "class PolynomialGradientDescentModel",
]

EXCLUDE_DEMO_TOKENS = [
    "plt.show(", "plt.plot(", "plt.scatter(", "Axes3D",
    "plot_surface(", "display(", "model =", "print("
]

def _load_code_cells_from_notebook(path):
    with io.open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            yield "".join(cell.get("source", []))

def _looks_like_def(src):
    return ("def " in src) or ("class " in src)

def _matches_includes(src):
    return any(pat in src for pat in INCLUDE_PATTERNS)

def _looks_like_demo(src):
    return any(tok in src for tok in EXCLUDE_DEMO_TOKENS)

def _exec_defs_only(namespace):
    had_syntax_error = False
    syntax_errors = []

    for src in _load_code_cells_from_notebook(NB_PATH):
        if _looks_like_def(src) and _matches_includes(src) and not _looks_like_demo(src):
            try:
                exec(src, namespace)
            except SyntaxError as e:
                had_syntax_error = True
                syntax_errors.append(str(e))

    if had_syntax_error:
        pytest.fail(
            "Notebook enthält unvollständige TODOs / Syntaxfehler in Definitionszellen.\n\n"
            + "\n".join(syntax_errors)
        )

# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture(scope="module")
def ns():
    """
    Namespace mit NUR den relevanten Definitionen + np.
    Falls mean_squared_error im Notebook nicht existiert, wird eine einfache
    Variante eingespeist (wird zur MSE-Historie im Modell benötigt).
    """
    namespace = {"np": np}
    _exec_defs_only(namespace)
    # Erwartete Symbole
    missing = [sym for sym in ("generate_sine_data", "PolynomialGradientDescentModel") if sym not in namespace]
    if missing:
        pytest.fail(f"Erwartete Symbole fehlen (TODOs nicht umgesetzt?): {missing}")
    # Fallback für mean_squared_error, falls nicht im Notebook definiert
    if "mean_squared_error" not in namespace:
        def _mse(y_true, y_pred):
            yt = np.asarray(y_true, dtype=float)
            yp = np.asarray(y_pred, dtype=float)
            return float(np.mean((yt - yp) ** 2))
        namespace["mean_squared_error"] = _mse
    return namespace

# ---------------------------
# Tests: generate_sine_data
# ---------------------------

def test_structure_and_exact_x(ns):
    gen = ns["generate_sine_data"]
    n = 200
    
    # Test: Überprüft, ob die Funktion Arrays der richtigen Form und Werte erzeugt.
    # Erwartung: x ist ein gleichmäßig verteiltes linspace zwischen 0 und 2*pi,
    #            y hat dieselbe Länge.
    x, y = gen(n, noise=0.5)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.shape == (n,) and y.shape == (n,)
    np.testing.assert_allclose(x, np.linspace(0, 2*np.pi, n))


def test_no_noise(ns):
    gen = ns["generate_sine_data"]

    # Test: Überprüft die Kernfunktionalität ohne Zufallseinfluss.
    # Erwartung: Wenn noise=0, dann ist y exakt sin(x).
    x, y = gen(100, noise=0)
    np.testing.assert_allclose(y, np.sin(x))


def test_reproducibility_with_seed(ns):
    gen = ns["generate_sine_data"]

    # Test: Stellt sicher, dass bei gleichem Zufalls-Seed dieselben Ergebnisse entstehen.
    # Erwartung: Zwei Aufrufe mit gleichem Seed liefern identische x- und y-Werte.
    np.random.seed(42)
    x1, y1 = gen(150, noise=0.2)
    np.random.seed(42)
    x2, y2 = gen(150, noise=0.2)

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)

# ---------------------------
# Hilfsfunktionen für Modell-Tests
# ---------------------------

def _mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))

def _numeric_grad(model, X, y, eps=1e-6):
    """
    Numerischer Gradient der MSE (zentrale Finite Differenzen)
    im selben globalen Namespace wie das Model.
    """
    base = model.weights.copy()
    grads = []
    for j in range(len(base)):
        w_plus = base.copy(); w_plus[j] += eps
        w_minus = base.copy(); w_minus[j] -= eps

        model.weights = w_plus
        m_plus = _mse(y, model.predict(X))

        model.weights = w_minus
        m_minus = _mse(y, model.predict(X))

        grads.append((m_plus - m_minus) / (2 * eps))
    model.weights = base
    return np.array(grads)

# ---------------------------
# Tests: PolynomialGradientDescentModel (5 Stück)
# ---------------------------

def test_predict_is_cubic_forward_pass(ns):
    """predict berechnet w0 + w1*x + w2*x^2 + w3*x^3 (vektorisiert)."""
    M = ns["PolynomialGradientDescentModel"]
    m = M()
    m.weights = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.array([0.0, 1.0, -2.0, 0.5])
    expected = 1 + 2*X + 3*(X**2) + 4*(X**3)
    y_hat = m.predict(X)
    assert isinstance(y_hat, np.ndarray)
    np.testing.assert_allclose(y_hat, expected, atol=1e-10)


def test_compute_gradient_closed_form(ns):
    """Analytischer Gradient entspricht der hergeleiteten Formel."""
    M = ns["PolynomialGradientDescentModel"]
    m = M()
    m.weights = np.array([0.3, -0.7, 0.2, 1.1])
    X = np.array([-2., -1., 0., 1., 2.])
    y = np.array([5.0, 0.5, -1.0, 0.0, 3.5])

    n = len(X)
    y_pred = m.predict(X)
    e = y - y_pred
    expected = np.array([
        (-2/n) * np.sum(e),
        (-2/n) * np.sum(X * e),
        (-2/n) * np.sum((X**2) * e),
        (-2/n) * np.sum((X**3) * e),
    ])
    got = np.array(m.compute_gradient(X, y))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_gradient_matches_numeric(ns):
    """Vergleich analytischer und numerischer Gradient."""
    rng = np.random.default_rng(0)
    M = ns["PolynomialGradientDescentModel"]
    m = M()
    m.weights = rng.normal(size=4)
    X = rng.normal(size=25)
    true_w = np.array([0.6, -0.4, 0.2, 0.8])
    y = true_w[0] + true_w[1]*X + true_w[2]*(X**2) + true_w[3]*(X**3) + rng.normal(0, 0.05, size=X.shape)

    analytic = np.array(m.compute_gradient(X, y))
    numeric = _numeric_grad(m, X, y)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_single_gradient_step_reduces_mse(ns):
    """Ein kleiner Schritt in -Gradientenrichtung reduziert den MSE (Vorzeichen korrekt)."""
    rng = np.random.default_rng(1)
    M = ns["PolynomialGradientDescentModel"]
    m = M()
    X = rng.uniform(-2, 2, size=120)
    w_true = np.array([1.0, -0.5, 0.25, 0.75])
    y = w_true[0] + w_true[1]*X + w_true[2]*(X**2) + w_true[3]*(X**3)

    mse_before = _mse(y, m.predict(X))
    grad = np.array(m.compute_gradient(X, y))
    m.weights = m.weights - 1e-3 * grad
    mse_after = _mse(y, m.predict(X))

    assert mse_after < mse_before, "Gradientenrichtung/Vorzeichen scheint falsch."


def test_fit_reduces_mse_over_iterations(ns):
    """Training über mehrere Iterationen sollte MSE klar reduzieren."""
    rng = np.random.default_rng(7)
    M = ns["PolynomialGradientDescentModel"]
    m = M(learning_rate=2e-3, n_iterations=800)

    X = np.linspace(-2, 2, 160)
    w_true = np.array([1.2, -0.5, 0.25, 0.75])
    y = w_true[0] + w_true[1]*X + w_true[2]*(X**2) + w_true[3]*(X**3) + rng.normal(0, 0.05, size=X.shape)

    mse_before = _mse(y, m.predict(X))
    m.fit(X, y)
    mse_after = _mse(y, m.predict(X))

    assert mse_after < mse_before * 0.6, "Training reduziert den Fehler nicht signifikant."
