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
# Tests: PolynomialGradientDescentModel
# ---------------------------

def _mse_local(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))

def test_poly_predict_forward_pass(ns):
    Model = ns["PolynomialGradientDescentModel"]
    m = Model()
    # Setze bekannte Gewichte: y = a + b x + c x^2 + d x^3
    a, b, c, d = 1.0, -2.0, 0.5, 0.1
    m.weights = np.array([a, b, c, d], dtype=float)
    X = np.array([0.0, 1.0, 2.0, -1.0])
    y_pred = m.predict(X)
    expected = a + b * X + c * X**2 + d * X**3
    assert np.allclose(y_pred, expected)

def test_poly_compute_gradient_matches_numeric(ns):
    Model = ns["PolynomialGradientDescentModel"]
    m = Model()
    # Erzeuge kleine Daten (ohne Rauschen) von einer bekannten Kubik
    true_w = np.array([0.3, -0.7, 0.2, 0.05], dtype=float)
    X = np.linspace(-2, 2, 25)
    y = true_w[0] + true_w[1]*X + true_w[2]*X**2 + true_w[3]*X**3

    # Starte bei anderen Gewichten, damit Gradienten ≠ 0
    m.weights = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # Analytische Gradienten aus compute_gradient
    g = np.array(m.compute_gradient(X, y), dtype=float)

    # Numerische Gradienten via finite differences auf MSE
    eps = 1e-6
    num_g = np.zeros_like(m.weights, dtype=float)
    base_mse = _mse_local(y, m.predict(X))
    for j in range(len(m.weights)):
        w_perturb = m.weights.copy()
        w_perturb[j] += eps
        # Temporär einsetzen
        old = m.weights[j]
        m.weights[j] = w_perturb[j]
        mse_plus = _mse_local(y, m.predict(X))
        m.weights[j] = old
        num_g[j] = (mse_plus - base_mse) / eps

    # compute_gradient sollte die MSE-Gradienten liefern
    assert np.allclose(g, num_g, rtol=1e-3, atol=1e-4)

def test_poly_fit_learns_and_mse_drops(ns):
    Model = ns["PolynomialGradientDescentModel"]
    m = Model(learning_rate=1e-3, n_iterations=2000)

    # Daten von bekannter Kubik mit geringem Rauschen
    rng = np.random.default_rng(123)
    true_w = np.array([1.5, -0.8, 0.3, 0.02], dtype=float)
    X = np.linspace(-3, 3, 120)
    y_clean = true_w[0] + true_w[1]*X + true_w[2]*X**2 + true_w[3]*X**3
    y = y_clean + rng.normal(0, 0.2, size=X.shape)

    m.fit(X, y)
    hist = m.get_mse_history()
    assert isinstance(hist, list) and len(hist) > 5
    assert hist[0] > hist[-1], "MSE sollte im Verlauf sinken."

    # Qualitätscheck: finale MSE signifikant unter Varianz des Rauschens
    final_mse = _mse_local(y, m.predict(X))
    assert final_mse < 0.2**2 * 2.0  # locker bemessen

