"""
symcalc.py — Calculadora simbólica con salida LaTeX y OMML
Uso: python symcalc.py

Requiere: pip install sympy
"""

from sympy import *
from sympy.printing.latex import latex as sym_latex
import textwrap

# ─── Símbolos predefinidos ──────────────────────────────────────────────────
t, s, z, x, y = symbols('t s z x y', real=True)
omega, nu = symbols('omega nu', real=True)
a, b, c, n, k = symbols('a b c n k', real=True, positive=True)
f, g = symbols('f g', cls=Function)

PREDEFINED = {
    't': t, 's': s, 'z': z, 'x': x, 'y': y,
    'omega': omega, 'nu': nu,
    'a': a, 'b': b, 'c': c, 'n': n, 'k': k,
}

# ─── Conversión básica SymPy → OMML ─────────────────────────────────────────
def _wrap_omml(inner: str) -> str:
    return (
        '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">\n'
        f'  {inner}\n'
        '</m:oMath>'
    )

def _omml_run(text: str) -> str:
    return f'<m:r><m:t>{text}</m:t></m:r>'

def _omml_fraction(num: str, den: str) -> str:
    return (
        f'<m:f>'
        f'<m:num>{_omml_run(num)}</m:num>'
        f'<m:den>{_omml_run(den)}</m:den>'
        f'</m:f>'
    )

def _omml_sup(base: str, exp: str) -> str:
    return (
        f'<m:sSup>'
        f'<m:e>{_omml_run(base)}</m:e>'
        f'<m:sup>{_omml_run(exp)}</m:sup>'
        f'</m:sSup>'
    )

def sympy_to_omml(expr) -> str:
    """
    Conversión heurística SymPy → OMML.
    Cubre los patrones más comunes en señales y óptica.
    Para expresiones muy complejas, devuelve OMML con LaTeX como fallback.
    """
    # Fracción simple: a/b
    if isinstance(expr, Mul):
        numer, denom = fraction(expr)
        if denom != 1:
            n_str = str(numer).replace('**', '^').replace('*', '·')
            d_str = str(denom).replace('**', '^').replace('*', '·')
            return _wrap_omml(_omml_fraction(n_str, d_str))

    # Potencia simple
    if isinstance(expr, Pow):
        base_, exp_ = expr.args
        return _wrap_omml(_omml_sup(str(base_), str(exp_)))

    # Fallback general: LaTeX embebido en OMML como texto plano
    latex_str = sym_latex(expr)
    return _wrap_omml(_omml_run(latex_str))


# ─── Transformada de Fourier con convenio angular ────────────────────────────
def _fourier_transform(expr):
    # F(ω) = ∫ f(t) e^{-iωt} dt  (convenio angular)
    # Los símbolos sin supuesto real bloquean la evaluación de integrales
    # distribucionales ∫ e^{i·a·t} dt = 2π·δ(a). Los hacemos reales
    # temporalmente y restauramos al final.
    free = expr.free_symbols - {t}
    to_real = {s: Symbol(s.name, real=True) for s in free if not s.is_real}
    expr_r = expr.subs(to_real)
    rev = {v: k for k, v in to_real.items()}

    for cand in (expr_r.rewrite(exp), expr_r):
        try:
            result = integrate(cand * exp(-I * omega * t), (t, -oo, oo))
            if not result.has(Integral):
                return result.subs(rev)
        except Exception:
            pass

    result = fourier_transform(expr_r, t, omega)
    if not result.has(FourierTransform):
        return result.subs(rev)

    return fourier_transform(expr, t, omega)


# ─── Operaciones disponibles ─────────────────────────────────────────────────
OPERATIONS = {
    '1': ('Transformada de Fourier', _fourier_transform),
    '2': ('Transformada de Laplace',
          lambda expr: laplace_transform(expr, t, s, noconds=True)),
    '3': ('Transformada Z (discreta)',
          lambda expr: z_transform(expr, n, z)),
    '4': ('Derivada respecto a t',
          lambda expr: diff(expr, t)),
    '5': ('Integral indefinida respecto a t',
          lambda expr: integrate(expr, t)),
    '6': ('Simplificar',
          lambda expr: simplify(expr)),
    '7': ('Expandir',
          lambda expr: expand(expr)),
    '8': ('Factorizar',
          lambda expr: factor(expr)),
    '9': ('Serie de Taylor en t=0 (orden 6)',
          lambda expr: series(expr, t, 0, 6).removeO()),
}

# ─── Helpers de visualización ────────────────────────────────────────────────
SEP = '─' * 60

def print_header():
    print()
    print('╔══════════════════════════════════════════════════════════╗')
    print('║          symcalc — Calculadora Simbólica (SymPy)         ║')
    print('║    Salida: pretty-print  ·  LaTeX  ·  OMML               ║')
    print('╚══════════════════════════════════════════════════════════╝')
    print()
    print('Símbolos predefinidos:')
    print('  Tiempo/dominio : t, x, y, s, z, n')
    print('  Frecuencia     : omega, nu')
    print('  Parámetros (+) : a, b, c, k')
    print('  Funciones esp. : exp(), sin(), cos(), Heaviside(), DiracDelta()')
    print()
    print('Ejemplos de expresiones:')
    print('  exp(-a*t) * Heaviside(t)')
    print('  sin(omega*t) / t')
    print('  exp(-a*t**2)')
    print('  1 / (a + t**2)')
    print()

def print_menu():
    print(SEP)
    print('Operaciones disponibles:')
    for key, (name, _) in OPERATIONS.items():
        print(f'  [{key}] {name}')
    print('  [0] Salir')
    print(SEP)

def print_result(operation_name: str, result):
    print()
    print(f'  Operación : {operation_name}')
    print(SEP)

    # Pretty-print
    print('  [Pretty-print]')
    pretty_str = pretty(result, use_unicode=True)
    for line in pretty_str.splitlines():
        print('    ' + line)
    print()

    # LaTeX
    latex_str = sym_latex(result)
    print('  [LaTeX]')
    print(f'    ${latex_str}$')
    print()

    # OMML
    omml_str = sympy_to_omml(result)
    print('  [OMML]')
    for line in omml_str.splitlines():
        print('    ' + line)
    print()


# ─── Bucle principal ─────────────────────────────────────────────────────────
def main():
    print_header()

    while True:
        # Entrada de expresión
        print()
        raw = input('  Expresión > ').strip()
        if not raw or raw == '0':
            print('\n  Hasta luego.\n')
            break

        # Parsear con sympify usando el namespace de símbolos predefinidos
        try:
            expr = sympify(raw, locals=PREDEFINED)
        except Exception as e:
            print(f'  ✗ Error al parsear la expresión: {e}')
            continue

        print()
        print('  Expresión reconocida:')
        pprint(expr, use_unicode=True)

        # Selección de operación
        print_menu()
        op = input('  Operación > ').strip()

        if op == '0':
            print('\n  Hasta luego.\n')
            break

        if op not in OPERATIONS:
            print('  ✗ Opción no válida.')
            continue

        op_name, op_func = OPERATIONS[op]

        try:
            result = op_func(expr)
            print_result(op_name, result)
        except Exception as e:
            print(f'\n  ✗ Error al calcular: {e}')
            print('  Puede que la operación no sea aplicable a esta expresión.')

        # Continuar o salir
        again = input('  ¿Otra operación? [s/N] > ').strip().lower()
        if again != 's':
            print('\n  Hasta luego.\n')
            break


if __name__ == '__main__':
    main()