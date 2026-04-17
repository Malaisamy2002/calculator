"""
╔══════════════════════════════════════════════════════════╗
║     Advanced Scientific & Graphing Calculator            ║
║     Features: Scientific, Graphing, Algebra, Calculus    ║
║     GUI: Tkinter | Plots: Matplotlib | Math: SymPy/NumPy ║
╚══════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sympy as sp
from sympy import symbols, solve, diff, integrate, factorial, simplify, latex
from sympy import sin as sp_sin, cos as sp_cos, tan as sp_tan
from sympy import log as sp_log, exp as sp_exp, sqrt as sp_sqrt, pi as sp_pi, E as sp_E
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import datetime
import json
import os
import re


# ─────────────────────────────────────────────────────────
#  THEME CONFIGURATION
# ─────────────────────────────────────────────────────────

THEMES = {
    "dark": {
        "bg":           "#0d1117",
        "surface":      "#161b22",
        "surface2":     "#21262d",
        "border":       "#30363d",
        "accent":       "#58a6ff",
        "accent2":      "#3fb950",
        "danger":       "#f85149",
        "warning":      "#d29922",
        "text":         "#e6edf3",
        "text_dim":     "#8b949e",
        "btn_num":      "#21262d",
        "btn_op":       "#1f3a5f",
        "btn_fn":       "#1a3a2a",
        "btn_eq":       "#1f6feb",
        "btn_clear":    "#5a1a1a",
        "btn_special":  "#2d2a1a",
        "display_bg":   "#010409",
        "display_fg":   "#58a6ff",
    },
    "light": {
        "bg":           "#f0f2f5",
        "surface":      "#ffffff",
        "surface2":     "#e8eaed",
        "border":       "#dadde1",
        "accent":       "#0066cc",
        "accent2":      "#1a7a40",
        "danger":       "#cc0000",
        "warning":      "#b36b00",
        "text":         "#1c1e21",
        "text_dim":     "#65676b",
        "btn_num":      "#ffffff",
        "btn_op":       "#cce0ff",
        "btn_fn":       "#ccf0d9",
        "btn_eq":       "#0066cc",
        "btn_clear":    "#ffcccc",
        "btn_special":  "#fff5cc",
        "display_bg":   "#f8f9fa",
        "display_fg":   "#0066cc",
    }
}


# ─────────────────────────────────────────────────────────
#  CALCULATOR ENGINE
# ─────────────────────────────────────────────────────────

class CalculatorEngine:
    """Core math engine – pure logic, no GUI."""

    def __init__(self):
        self.angle_mode = "DEG"   # DEG or RAD
        self.history: list[dict] = []
        self.memory: float = 0.0
        self.x = symbols('x')
        self._transformations = (standard_transformations +
                                  (implicit_multiplication_application,))

    # ── Angle helpers ──────────────────────────────────────
    def _to_rad(self, v: float) -> float:
        return math.radians(v) if self.angle_mode == "DEG" else v

    def _from_rad(self, v: float) -> float:
        return math.degrees(v) if self.angle_mode == "DEG" else v

    # ── Safe expression evaluator ──────────────────────────
    def evaluate(self, expr: str) -> str:
        """Evaluate a plain numeric expression string safely."""
        try:
            # Replace common tokens
            expr = expr.replace("^", "**").replace("π", "pi").replace("÷", "/").replace("×", "*")
            # Patch trig so they respect angle mode
            am = self.angle_mode
            safe_globals = {
                "__builtins__": {},
                "pi": math.pi, "e": math.e,
                "sin":   lambda v: math.sin(self._to_rad(v)),
                "cos":   lambda v: math.cos(self._to_rad(v)),
                "tan":   lambda v: math.tan(self._to_rad(v)),
                "asin":  lambda v: self._from_rad(math.asin(v)),
                "acos":  lambda v: self._from_rad(math.acos(v)),
                "atan":  lambda v: self._from_rad(math.atan(v)),
                "sinh":  math.sinh, "cosh": math.cosh, "tanh": math.tanh,
                "log":   math.log10,
                "ln":    math.log,
                "log2":  math.log2,
                "sqrt":  math.sqrt,
                "cbrt":  lambda v: v ** (1/3) if v >= 0 else -((-v) ** (1/3)),
                "abs":   abs,
                "ceil":  math.ceil,
                "floor": math.floor,
                "round": round,
                "factorial": math.factorial,
                "exp":   math.exp,
                "pow":   pow,
            }
            result = eval(compile(expr, "<string>", "eval"), safe_globals)
            result = float(result)
            result_str = self._format_number(result)
            self._add_history(expr, result_str)
            return result_str
        except ZeroDivisionError:
            return "Error: Division by zero"
        except OverflowError:
            return "Error: Overflow"
        except Exception as e:
            return f"Error: {e}"

    def _format_number(self, n: float) -> str:
        if n == int(n) and abs(n) < 1e15:
            return str(int(n))
        if abs(n) > 1e10 or (abs(n) < 1e-4 and n != 0):
            return f"{n:.6e}"
        return f"{n:.10g}"

    # ── History ────────────────────────────────────────────
    def _add_history(self, expr: str, result: str):
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "expression": expr,
            "result": result,
            "angle_mode": self.angle_mode,
        }
        self.history.append(entry)

    # ── Memory ─────────────────────────────────────────────
    def memory_store(self, value: float): self.memory = value
    def memory_recall(self) -> float:    return self.memory
    def memory_add(self, value: float):  self.memory += value
    def memory_clear(self):              self.memory = 0.0

    # ── Equation Solver ────────────────────────────────────
    def solve_equation(self, eq_str: str, var: str = "x") -> str:
        """Solve f(x) = 0  or  lhs = rhs  equations."""
        try:
            sym_var = symbols(var)
            eq_str = eq_str.replace("^", "**").replace("π", "pi")
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=", 1)
                equation = parse_expr(lhs, transformations=self._transformations) - \
                           parse_expr(rhs, transformations=self._transformations)
            else:
                equation = parse_expr(eq_str, transformations=self._transformations)
            solutions = solve(equation, sym_var)
            if not solutions:
                return "No real solutions found."
            sol_strs = [str(simplify(s)) for s in solutions]
            result = f"{var} = {', '.join(sol_strs)}"
            self._add_history(f"solve({eq_str})", result)
            return result
        except Exception as e:
            return f"Error: {e}"

    # ── Derivative ─────────────────────────────────────────
    def derivative(self, expr_str: str, var: str = "x", order: int = 1) -> str:
        try:
            sym_var = symbols(var)
            expr_str = expr_str.replace("^", "**").replace("π", "pi")
            expr = parse_expr(expr_str, transformations=self._transformations)
            result = diff(expr, sym_var, order)
            result = simplify(result)
            result_str = str(result)
            self._add_history(f"d/d{var}({expr_str})", result_str)
            return result_str
        except Exception as e:
            return f"Error: {e}"

    # ── Integral ───────────────────────────────────────────
    def integral(self, expr_str: str, var: str = "x",
                  lower=None, upper=None) -> str:
        try:
            sym_var = symbols(var)
            expr_str = expr_str.replace("^", "**").replace("π", "pi")
            expr = parse_expr(expr_str, transformations=self._transformations)
            if lower is not None and upper is not None:
                result = integrate(expr, (sym_var,
                                           parse_expr(str(lower)),
                                           parse_expr(str(upper))))
                result_str = f"{simplify(result)}"
            else:
                result = integrate(expr, sym_var)
                result_str = f"{simplify(result)} + C"
            self._add_history(f"∫({expr_str})d{var}", result_str)
            return result_str
        except Exception as e:
            return f"Error: {e}"

    # ── Matrix Operations ──────────────────────────────────
    def matrix_operation(self, mat_a: list, mat_b: list, op: str) -> str:
        try:
            A = np.array(mat_a, dtype=float)
            B = np.array(mat_b, dtype=float)
            if op == "add":
                R = A + B
            elif op == "sub":
                R = A - B
            elif op == "mul":
                R = A @ B
            elif op == "det_a":
                return f"det(A) = {np.linalg.det(A):.6g}"
            elif op == "inv_a":
                R = np.linalg.inv(A)
            else:
                return "Unknown operation"
            return "\n".join(["  ".join([f"{v:.4g}" for v in row]) for row in R])
        except Exception as e:
            return f"Error: {e}"

    # ── Unit Conversions ───────────────────────────────────
    CONVERSIONS = {
        # length → metres
        "length": {
            "m": 1, "km": 1e3, "cm": 1e-2, "mm": 1e-3,
            "mi": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254,
        },
        # weight → kg
        "weight": {
            "kg": 1, "g": 1e-3, "mg": 1e-6, "lb": 0.453592, "oz": 0.0283495, "t": 1e3,
        },
    }

    def convert_unit(self, value: float, from_unit: str, to_unit: str) -> str:
        for category, units in self.CONVERSIONS.items():
            if from_unit in units and to_unit in units:
                result = value * units[from_unit] / units[to_unit]
                r = f"{value} {from_unit} = {self._format_number(result)} {to_unit}"
                self._add_history(f"convert({value} {from_unit} → {to_unit})", r)
                return r
        # Temperature
        temps = {"C", "F", "K"}
        if from_unit in temps and to_unit in temps:
            return self._temp_convert(value, from_unit, to_unit)
        return "Unsupported conversion"

    def _temp_convert(self, v, frm, to) -> str:
        to_c = {"C": v, "F": (v - 32) * 5/9, "K": v - 273.15}
        from_c = {"C": lambda c: c, "F": lambda c: c * 9/5 + 32, "K": lambda c: c + 273.15}
        c = to_c[frm]
        result = from_c[to](c)
        r = f"{v}°{frm} = {self._format_number(result)}°{to}"
        self._add_history(f"temp({v}°{frm} → °{to})", r)
        return r

    def angle_convert(self, value: float, frm: str, to: str) -> str:
        if frm == "DEG" and to == "RAD":
            r = math.radians(value)
        elif frm == "RAD" and to == "DEG":
            r = math.degrees(value)
        else:
            r = value
        result = f"{value}° ({frm}) = {self._format_number(r)} ({to})"
        self._add_history(f"angle_convert({value} {frm}→{to})", result)
        return result

    # ── Save History ───────────────────────────────────────
    def save_history(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)

    def export_history_txt(self, filepath: str):
        with open(filepath, "w") as f:
            f.write("═" * 60 + "\n")
            f.write("  CALCULATOR HISTORY\n")
            f.write("═" * 60 + "\n\n")
            for entry in self.history:
                f.write(f"[{entry['timestamp']}]  ({entry['angle_mode']})\n")
                f.write(f"  {entry['expression']}  =  {entry['result']}\n\n")


# ─────────────────────────────────────────────────────────
#  GRAPHING ENGINE
# ─────────────────────────────────────────────────────────

class GraphingEngine:
    """Handles function plotting inside a Tk window."""

    def __init__(self, parent, theme: dict):
        self.parent = parent
        self.theme = theme
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.plots: list[dict] = []    # {label, xs, ys, color}
        self.color_cycle = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
                             "#ffa657", "#79c0ff", "#56d364"]
        self._color_idx = 0

    def _next_color(self) -> str:
        c = self.color_cycle[self._color_idx % len(self.color_cycle)]
        self._color_idx += 1
        return c

    def _get_safe_fn(self, expr_str: str):
        """Compile an expression of x into a vectorised NumPy function."""
        expr_str = (expr_str
                    .replace("^", "**")
                    .replace("π", "str(np.pi)")
                    .replace("pi", "np.pi")
                    .replace("sin(", "np.sin(")
                    .replace("cos(", "np.cos(")
                    .replace("tan(", "np.tan(")
                    .replace("asin(", "np.arcsin(")
                    .replace("acos(", "np.arccos(")
                    .replace("atan(", "np.arctan(")
                    .replace("sqrt(", "np.sqrt(")
                    .replace("exp(", "np.exp(")
                    .replace("log(", "np.log10(")
                    .replace("ln(", "np.log(")
                    .replace("abs(", "np.abs(")
                    .replace("e", "np.e")   # last so it doesn't mangle "exp"
                    )
        # Re-fix "np.exp" that got broken by the e→np.e pass
        expr_str = expr_str.replace("np.np.e", "np.e")
        ns = {"np": np, "__builtins__": {}}
        code = compile(expr_str, "<graph>", "eval")
        def fn(x_arr):
            ns["x"] = x_arr
            return eval(code, ns)
        return fn

    def add_plot(self, expr_str: str, x_min: float = -10, x_max: float = 10,
                  label: str | None = None) -> str:
        try:
            fn = self._get_safe_fn(expr_str)
            xs = np.linspace(x_min, x_max, 2000)
            ys = fn(xs)
            ys = np.where(np.abs(ys) > 1e10, np.nan, ys)
            color = self._next_color()
            lbl = label or f"y = {expr_str}"
            self.plots.append({"label": lbl, "xs": xs, "ys": ys, "color": color})
            return "OK"
        except Exception as e:
            return f"Error: {e}"

    def clear_plots(self):
        self.plots.clear()
        self._color_idx = 0

    def render(self, frame: tk.Frame):
        """Draw / redraw into `frame`."""
        t = self.theme
        bg = t["bg"]
        fg = t["text"]

        # Destroy old canvas
        for w in frame.winfo_children():
            w.destroy()

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor=bg)
        self.ax  = self.fig.add_subplot(111, facecolor=t["surface"])
        self.ax.tick_params(colors=fg, labelcolor=fg)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(t["border"])

        for p in self.plots:
            self.ax.plot(p["xs"], p["ys"], color=p["color"],
                         linewidth=2, label=p["label"])

        self.ax.axhline(0, color=t["text_dim"], linewidth=0.8, linestyle="--")
        self.ax.axvline(0, color=t["text_dim"], linewidth=0.8, linestyle="--")
        self.ax.grid(True, color=t["border"], alpha=0.5, linestyle=":")
        if self.plots:
            legend = self.ax.legend(facecolor=t["surface2"],
                                     edgecolor=t["border"],
                                     labelcolor=fg, fontsize=9)
        self.ax.set_xlabel("x", color=fg)
        self.ax.set_ylabel("y", color=fg)
        self.ax.set_title("Function Plot", color=fg, fontsize=11, pad=8)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, frame)
        self.toolbar.update()
        self.toolbar.config(background=t["surface2"])
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# ─────────────────────────────────────────────────────────
#  MAIN GUI
# ─────────────────────────────────────────────────────────

class ScientificCalculatorApp:
    """Main Tkinter application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Advanced Scientific Calculator")
        self.root.resizable(True, True)
        self.root.minsize(900, 640)

        self.theme_name = tk.StringVar(value="dark")
        self.engine = CalculatorEngine()
        self.graphing = None          # initialised after theme is set
        self.expression = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="0")
        self.angle_mode_var = tk.StringVar(value="DEG")
        self._build_ui()

    # ── Theme ──────────────────────────────────────────────
    @property
    def T(self) -> dict:
        return THEMES[self.theme_name.get()]

    def _apply_theme(self):
        t = self.T
        self.root.config(bg=t["bg"])
        # Recolour all widgets
        self._colour_widget(self.root, t)
        # Redraw graph if open
        if hasattr(self, "_graph_frame") and self.graphing:
            self.graphing.theme = t
            self.graphing.render(self._graph_frame)

    def _colour_widget(self, widget, t: dict):
        cls = widget.winfo_class()
        try:
            if cls in ("Frame", "Labelframe", "TFrame"):
                widget.config(bg=t["surface"])
            elif cls == "Label":
                widget.config(bg=widget.master.cget("bg") if widget.master else t["surface"],
                               fg=t["text"])
            elif cls == "Button":
                pass  # handled individually
        except Exception:
            pass
        for child in widget.winfo_children():
            self._colour_widget(child, t)

    # ── UI Builder ─────────────────────────────────────────
    def _build_ui(self):
        t = self.T
        self.root.config(bg=t["bg"])

        # ── Top bar ──
        top = tk.Frame(self.root, bg=t["surface"], pady=6)
        top.pack(fill=tk.X, side=tk.TOP)
        tk.Label(top, text="⊕  Advanced Calculator",
                 bg=t["surface"], fg=t["accent"],
                 font=("Courier New", 14, "bold")).pack(side=tk.LEFT, padx=14)

        # Theme toggle
        tk.Button(top, text="☀ Light" if self.theme_name.get()=="dark" else "☾ Dark",
                  command=self._toggle_theme,
                  bg=t["btn_special"], fg=t["text"],
                  font=("Courier New", 9), relief="flat",
                  activebackground=t["accent"], cursor="hand2",
                  bd=0, padx=8, pady=4).pack(side=tk.RIGHT, padx=10)
        self._theme_btn_ref = top.winfo_children()[-1]

        # Angle mode
        for mode in ("DEG", "RAD"):
            tk.Radiobutton(top, text=mode, variable=self.angle_mode_var,
                           value=mode, command=self._set_angle_mode,
                           bg=t["surface"], fg=t["text_dim"],
                           selectcolor=t["surface2"],
                           activebackground=t["surface"],
                           font=("Courier New", 9, "bold")).pack(side=tk.RIGHT, padx=4)
        tk.Label(top, text="Mode:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.RIGHT, padx=2)

        # ── Notebook tabs ──
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=t["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",
                         background=t["surface2"], foreground=t["text_dim"],
                         padding=[14, 6], font=("Courier New", 9, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", t["surface"])],
                  foreground=[("selected", t["accent"])])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self._build_calculator_tab()
        self._build_graphing_tab()
        self._build_algebra_tab()
        self._build_matrix_tab()
        self._build_conversion_tab()
        self._build_history_tab()

    # ── Calculator Tab ─────────────────────────────────────
    def _build_calculator_tab(self):
        t = self.T
        frame = tk.Frame(self.notebook, bg=t["bg"])
        self.notebook.add(frame, text="  🔢 Calculator  ")

        # Display area
        disp_frame = tk.Frame(frame, bg=t["display_bg"], bd=0,
                               relief="flat", padx=12, pady=10)
        disp_frame.pack(fill=tk.X, padx=12, pady=(12, 4))

        # Expression line
        self._expr_label = tk.Label(disp_frame, textvariable=self.expression,
                                    anchor="e", bg=t["display_bg"],
                                    fg=t["text_dim"],
                                    font=("Courier New", 11))
        self._expr_label.pack(fill=tk.X)

        # Result line
        self._result_label = tk.Label(disp_frame, textvariable=self.result_var,
                                       anchor="e", bg=t["display_bg"],
                                       fg=t["display_fg"],
                                       font=("Courier New", 28, "bold"))
        self._result_label.pack(fill=tk.X)

        # Memory label
        self._mem_label = tk.Label(disp_frame, text="M: 0",
                                    anchor="w", bg=t["display_bg"],
                                    fg=t["text_dim"],
                                    font=("Courier New", 9))
        self._mem_label.pack(fill=tk.X)

        # Button grid
        btn_area = tk.Frame(frame, bg=t["bg"])
        btn_area.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._btn_refs = []
        self._make_buttons(btn_area)

    def _make_buttons(self, parent):
        t = self.T

        # Layout: (label, colspan, type, command_arg)
        # types: num, op, fn, eq, clear, special
        rows = [
            [("MC", 1, "special", "MC"), ("MR", 1, "special", "MR"),
             ("M+", 1, "special", "M+"), ("MS", 1, "special", "MS"),
             ("(", 1, "op", "("),        (")", 1, "op", ")")],

            [("sin", 1, "fn", "sin("), ("cos", 1, "fn", "cos("),
             ("tan", 1, "fn", "tan("), ("ln",  1, "fn", "ln("),
             ("log", 1, "fn", "log("), ("√",   1, "fn", "sqrt(")],

            [("asin", 1, "fn", "asin("), ("acos", 1, "fn", "acos("),
             ("atan", 1, "fn", "atan("), ("x²",  1, "fn", "**2"),
             ("x³",  1, "fn", "**3"),    ("xʸ",  1, "fn", "**")],

            [("π",   1, "fn",  "π"),  ("e",  1, "fn", "e"),
             ("n!",  1, "fn", "factorial("), ("cbrt", 1, "fn", "cbrt("),
             ("exp", 1, "fn", "exp("),  ("%",   1, "op",  "%")],

            [("C",   1, "clear", "CLEAR"), ("⌫", 1, "clear", "BACK"),
             ("±",   1, "special", "NEG"), ("÷", 1, "op",  "/"),
             ("×",   1, "op",  "*"),       ("−", 1, "op",  "-")],

            [("7",  1, "num", "7"), ("8", 1, "num", "8"), ("9", 1, "num", "9"),
             ("+",  1, "op",  "+"), ("ANS", 1, "special", "ANS"), ("CE", 1, "clear", "CE")],

            [("4",  1, "num", "4"), ("5", 1, "num", "5"), ("6", 1, "num", "6"),
             ("1/x",1, "fn",  "1/"), ("x!", 1, "fn", "factorial("), ("←", 1, "special", "BACK")],

            [("1",  1, "num", "1"), ("2", 1, "num", "2"), ("3", 1, "num", "3"),
             (".",  1, "num", "."), ("00", 1, "num", "00"), ("=", 2, "eq", "=")],

            [("0",  2, "num", "0"), ("EXP", 1, "fn", "e**"), ("^", 1, "op", "**"), None],
        ]

        color_map = {
            "num":     (t["btn_num"],    t["text"]),
            "op":      (t["btn_op"],     t["accent"]),
            "fn":      (t["btn_fn"],     t["accent2"]),
            "eq":      (t["btn_eq"],     "#ffffff"),
            "clear":   (t["btn_clear"],  t["danger"]),
            "special": (t["btn_special"],t["warning"]),
        }

        for r_idx, row in enumerate(rows):
            col = 0
            for item in row:
                if item is None:
                    col += 1
                    continue
                label, span, btype, arg = item
                bg, fg = color_map.get(btype, (t["btn_num"], t["text"]))
                btn = tk.Button(
                    parent, text=label,
                    bg=bg, fg=fg,
                    font=("Courier New", 11, "bold"),
                    relief="flat", bd=0,
                    cursor="hand2",
                    activebackground=t["accent"],
                    activeforeground="#ffffff",
                    width=5, height=2,
                    command=lambda a=arg: self._btn_press(a)
                )
                btn.grid(row=r_idx, column=col, columnspan=span,
                         padx=2, pady=2, sticky="nsew")
                self._btn_refs.append(btn)
                col += span

        for r in range(len(rows)):
            parent.rowconfigure(r, weight=1)
        for c in range(6):
            parent.columnconfigure(c, weight=1)

    def _btn_press(self, arg: str):
        t = self.T
        current = self.expression.get()
        special = {"CLEAR", "BACK", "=", "NEG", "ANS",
                   "MC", "MR", "M+", "MS", "CE"}

        if arg == "CLEAR":
            self.expression.set("")
            self.result_var.set("0")
        elif arg in ("BACK", "CE"):
            self.expression.set(current[:-1])
        elif arg == "=":
            expr = self.expression.get()
            if expr:
                result = self.engine.evaluate(expr)
                self.result_var.set(result)
                if not result.startswith("Error"):
                    self.expression.set(expr + " =")
                    self._last_ans = result
        elif arg == "NEG":
            if current:
                self.expression.set(f"-({current})")
        elif arg == "ANS":
            ans = getattr(self, "_last_ans", "0")
            self.expression.set(current + ans)
        elif arg == "MC":
            self.engine.memory_clear()
            self._mem_label.config(text="M: 0")
        elif arg == "MR":
            self.expression.set(current + str(self.engine.memory_recall()))
        elif arg == "M+":
            try:
                self.engine.memory_add(float(self.result_var.get()))
                self._mem_label.config(text=f"M: {self.engine.memory}")
            except Exception:
                pass
        elif arg == "MS":
            try:
                self.engine.memory_store(float(self.result_var.get()))
                self._mem_label.config(text=f"M: {self.engine.memory}")
            except Exception:
                pass
        else:
            # Append to expression
            if self.expression.get().endswith(" ="):
                self.expression.set(arg)
            else:
                self.expression.set(current + arg)

    def _set_angle_mode(self):
        self.engine.angle_mode = self.angle_mode_var.get()

    # ── Graphing Tab ───────────────────────────────────────
    def _build_graphing_tab(self):
        t = self.T
        frame = tk.Frame(self.notebook, bg=t["bg"])
        self.notebook.add(frame, text="  📈 Graphing  ")
        self.graphing = GraphingEngine(frame, t)

        # Controls
        ctrl = tk.Frame(frame, bg=t["surface"], pady=6)
        ctrl.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(ctrl, text="f(x) =", bg=t["surface"],
                 fg=t["text"], font=("Courier New", 11, "bold")).pack(side=tk.LEFT, padx=6)

        self._graph_expr = tk.Entry(ctrl, font=("Courier New", 12),
                                     bg=t["display_bg"], fg=t["display_fg"],
                                     insertbackground=t["accent"],
                                     relief="flat", bd=4, width=28)
        self._graph_expr.pack(side=tk.LEFT, padx=4)
        self._graph_expr.insert(0, "sin(x)")

        tk.Label(ctrl, text="x:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=(12,2))
        self._xmin = tk.Entry(ctrl, width=6, font=("Courier New", 10),
                               bg=t["display_bg"], fg=t["text"],
                               insertbackground=t["accent"], relief="flat", bd=4)
        self._xmin.insert(0, "-10")
        self._xmin.pack(side=tk.LEFT, padx=2)

        tk.Label(ctrl, text="to", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT)
        self._xmax = tk.Entry(ctrl, width=6, font=("Courier New", 10),
                               bg=t["display_bg"], fg=t["text"],
                               insertbackground=t["accent"], relief="flat", bd=4)
        self._xmax.insert(0, "10")
        self._xmax.pack(side=tk.LEFT, padx=2)

        self._make_btn(ctrl, "Plot", self._do_plot, "eq").pack(side=tk.LEFT, padx=8)
        self._make_btn(ctrl, "Clear All", self._clear_plots, "clear").pack(side=tk.LEFT, padx=2)

        # Graph area
        self._graph_frame = tk.Frame(frame, bg=t["bg"])
        self._graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self.graphing.render(self._graph_frame)

        # Quick examples
        eg_frame = tk.Frame(frame, bg=t["surface"], pady=4)
        eg_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(eg_frame, text="Quick:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=8)
        for eg in ["sin(x)", "cos(x)", "x**2", "x**3-3*x", "tan(x)", "1/x", "exp(-x**2)"]:
            tk.Button(eg_frame, text=eg,
                      bg=t["btn_fn"], fg=t["accent2"],
                      font=("Courier New", 8), relief="flat", bd=0,
                      cursor="hand2",
                      command=lambda e=eg: self._quick_plot(e),
                      padx=6, pady=2).pack(side=tk.LEFT, padx=2)

    def _do_plot(self):
        expr = self._graph_expr.get().strip()
        try:
            xmin = float(self._xmin.get())
            xmax = float(self._xmax.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid x range")
            return
        result = self.graphing.add_plot(expr, xmin, xmax)
        if result != "OK":
            messagebox.showerror("Plot Error", result)
            return
        self.graphing.render(self._graph_frame)

    def _clear_plots(self):
        self.graphing.clear_plots()
        self.graphing.render(self._graph_frame)

    def _quick_plot(self, expr: str):
        self._graph_expr.delete(0, tk.END)
        self._graph_expr.insert(0, expr)
        self._do_plot()

    # ── Algebra Tab ────────────────────────────────────────
    def _build_algebra_tab(self):
        t = self.T
        frame = tk.Frame(self.notebook, bg=t["bg"])
        self.notebook.add(frame, text="  ∫ Algebra & Calculus  ")

        sections = [
            ("🔍 Equation Solver", self._build_solver_section),
            ("d/dx  Derivative",   self._build_derivative_section),
            ("∫  Integral",        self._build_integral_section),
        ]

        for title, builder in sections:
            sec = tk.LabelFrame(frame, text=f"  {title}  ",
                                 bg=t["surface"], fg=t["accent"],
                                 font=("Courier New", 10, "bold"),
                                 bd=1, relief="groove",
                                 labelanchor="nw")
            sec.pack(fill=tk.X, padx=14, pady=8)
            builder(sec)

    def _build_solver_section(self, parent):
        t = self.T
        row = tk.Frame(parent, bg=t["surface"])
        row.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(row, text="Equation:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT)
        self._eq_entry = tk.Entry(row, font=("Courier New", 12),
                                   bg=t["display_bg"], fg=t["display_fg"],
                                   insertbackground=t["accent"],
                                   relief="flat", bd=4, width=30)
        self._eq_entry.insert(0, "x**2 - 5*x + 6 = 0")
        self._eq_entry.pack(side=tk.LEFT, padx=6)

        tk.Label(row, text="Var:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)
        self._eq_var = tk.Entry(row, width=4, font=("Courier New", 10),
                                 bg=t["display_bg"], fg=t["text"],
                                 insertbackground=t["accent"], relief="flat", bd=4)
        self._eq_var.insert(0, "x")
        self._eq_var.pack(side=tk.LEFT, padx=4)

        self._make_btn(row, "Solve", self._do_solve, "eq").pack(side=tk.LEFT, padx=8)
        self._solve_result = tk.Label(parent, text="", bg=t["surface"],
                                       fg=t["accent2"], font=("Courier New", 11, "bold"))
        self._solve_result.pack(anchor="w", padx=10, pady=4)

    def _do_solve(self):
        eq = self._eq_entry.get().strip()
        var = self._eq_var.get().strip() or "x"
        result = self.engine.solve_equation(eq, var)
        self._solve_result.config(text=result)
        self._refresh_history()

    def _build_derivative_section(self, parent):
        t = self.T
        row = tk.Frame(parent, bg=t["surface"])
        row.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(row, text="f(x) =", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT)
        self._deriv_entry = tk.Entry(row, font=("Courier New", 12),
                                      bg=t["display_bg"], fg=t["display_fg"],
                                      insertbackground=t["accent"],
                                      relief="flat", bd=4, width=28)
        self._deriv_entry.insert(0, "x**3 + sin(x)")
        self._deriv_entry.pack(side=tk.LEFT, padx=6)

        tk.Label(row, text="Order:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)
        self._deriv_order = tk.Entry(row, width=3, font=("Courier New", 10),
                                      bg=t["display_bg"], fg=t["text"],
                                      insertbackground=t["accent"], relief="flat", bd=4)
        self._deriv_order.insert(0, "1")
        self._deriv_order.pack(side=tk.LEFT, padx=4)

        self._make_btn(row, "Differentiate", self._do_derivative, "fn").pack(side=tk.LEFT, padx=8)
        self._deriv_result = tk.Label(parent, text="", bg=t["surface"],
                                       fg=t["accent2"], font=("Courier New", 11, "bold"))
        self._deriv_result.pack(anchor="w", padx=10, pady=4)

    def _do_derivative(self):
        expr = self._deriv_entry.get().strip()
        try:
            order = int(self._deriv_order.get())
        except ValueError:
            order = 1
        result = self.engine.derivative(expr, order=order)
        self._deriv_result.config(text=f"f'(x) = {result}")
        self._refresh_history()

    def _build_integral_section(self, parent):
        t = self.T
        row = tk.Frame(parent, bg=t["surface"])
        row.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(row, text="f(x) =", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT)
        self._integ_entry = tk.Entry(row, font=("Courier New", 12),
                                      bg=t["display_bg"], fg=t["display_fg"],
                                      insertbackground=t["accent"],
                                      relief="flat", bd=4, width=24)
        self._integ_entry.insert(0, "x**2 + cos(x)")
        self._integ_entry.pack(side=tk.LEFT, padx=6)

        tk.Label(row, text="From:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)
        self._integ_lower = tk.Entry(row, width=5, font=("Courier New", 10),
                                      bg=t["display_bg"], fg=t["text"],
                                      insertbackground=t["accent"], relief="flat", bd=4)
        self._integ_lower.pack(side=tk.LEFT, padx=2)

        tk.Label(row, text="To:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)
        self._integ_upper = tk.Entry(row, width=5, font=("Courier New", 10),
                                      bg=t["display_bg"], fg=t["text"],
                                      insertbackground=t["accent"], relief="flat", bd=4)
        self._integ_upper.pack(side=tk.LEFT, padx=2)

        self._make_btn(row, "Integrate", self._do_integral, "fn").pack(side=tk.LEFT, padx=8)
        self._integ_result = tk.Label(parent, text="", bg=t["surface"],
                                       fg=t["accent2"], font=("Courier New", 11, "bold"))
        self._integ_result.pack(anchor="w", padx=10, pady=4)

    def _do_integral(self):
        expr = self._integ_entry.get().strip()
        lower = self._integ_lower.get().strip() or None
        upper = self._integ_upper.get().strip() or None
        result = self.engine.integral(expr, lower=lower, upper=upper)
        self._integ_result.config(text=f"∫f dx = {result}")
        self._refresh_history()

    # ── Matrix Tab ─────────────────────────────────────────
    def _build_matrix_tab(self):
        t = self.T
        frame = tk.Frame(self.notebook, bg=t["bg"])
        self.notebook.add(frame, text="  [ ] Matrix  ")

        def matrix_frame(parent, title, entries_var):
            sec = tk.LabelFrame(parent, text=f"  {title}  ",
                                 bg=t["surface"], fg=t["accent"],
                                 font=("Courier New", 9, "bold"),
                                 bd=1, relief="groove")
            sec.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
            grid = tk.Frame(sec, bg=t["surface"])
            grid.pack(padx=8, pady=8)
            entries = []
            for r in range(3):
                row_entries = []
                for c in range(3):
                    e = tk.Entry(grid, width=6, font=("Courier New", 10),
                                  bg=t["display_bg"], fg=t["display_fg"],
                                  insertbackground=t["accent"],
                                  justify="center", relief="flat", bd=3)
                    e.insert(0, "0")
                    e.grid(row=r, column=c, padx=2, pady=2)
                    row_entries.append(e)
                entries.append(row_entries)
            entries_var.append(entries)
            return sec

        matrices_row = tk.Frame(frame, bg=t["bg"])
        matrices_row.pack(fill=tk.X)
        self._mat_a_entries = []
        self._mat_b_entries = []
        matrix_frame(matrices_row, "Matrix A", self._mat_a_entries)
        matrix_frame(matrices_row, "Matrix B", self._mat_b_entries)

        # Extract first list because we used append trick
        self._mat_a_entries = self._mat_a_entries[0]
        self._mat_b_entries = self._mat_b_entries[0]

        ops_frame = tk.Frame(frame, bg=t["surface"], pady=8)
        ops_frame.pack(fill=tk.X, padx=14)

        tk.Label(ops_frame, text="Operation:", bg=t["surface"],
                 fg=t["text"], font=("Courier New", 10, "bold")).pack(side=tk.LEFT, padx=8)

        for op, lbl in [("add","A + B"), ("sub","A − B"), ("mul","A × B"),
                         ("det_a","det(A)"), ("inv_a","A⁻¹")]:
            self._make_btn(ops_frame, lbl,
                           lambda o=op: self._do_matrix(o),
                           "fn" if op not in ("det_a","inv_a") else "special"
                           ).pack(side=tk.LEFT, padx=4)

        self._mat_result = scrolledtext.ScrolledText(
            frame, height=6, font=("Courier New", 11),
            bg=t["display_bg"], fg=t["accent2"],
            relief="flat", bd=4)
        self._mat_result.pack(fill=tk.X, padx=14, pady=10)

    def _get_matrix(self, entries) -> list:
        mat = []
        for row in entries:
            mat.append([float(e.get() or 0) for e in row])
        return mat

    def _do_matrix(self, op: str):
        A = self._get_matrix(self._mat_a_entries)
        B = self._get_matrix(self._mat_b_entries)
        result = self.engine.matrix_operation(A, B, op)
        self._mat_result.delete("1.0", tk.END)
        self._mat_result.insert(tk.END, result)

    # ── Conversion Tab ─────────────────────────────────────
    def _build_conversion_tab(self):
        t = self.T
        frame = tk.Frame(self.notebook, bg=t["bg"])
        self.notebook.add(frame, text="  ⟺ Conversions  ")

        # ── Unit conversion ──
        sec1 = tk.LabelFrame(frame, text="  Unit Conversion  ",
                              bg=t["surface"], fg=t["accent"],
                              font=("Courier New", 10, "bold"),
                              bd=1, relief="groove")
        sec1.pack(fill=tk.X, padx=14, pady=10)

        row1 = tk.Frame(sec1, bg=t["surface"])
        row1.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(row1, text="Value:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT)
        self._conv_val = tk.Entry(row1, width=12, font=("Courier New", 12),
                                   bg=t["display_bg"], fg=t["display_fg"],
                                   insertbackground=t["accent"], relief="flat", bd=4)
        self._conv_val.insert(0, "100")
        self._conv_val.pack(side=tk.LEFT, padx=6)

        all_units = (list(CalculatorEngine.CONVERSIONS["length"].keys()) +
                     list(CalculatorEngine.CONVERSIONS["weight"].keys()) +
                     ["C", "F", "K"])

        tk.Label(row1, text="From:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=6)
        self._conv_from = ttk.Combobox(row1, values=all_units, width=6,
                                        font=("Courier New", 10))
        self._conv_from.set("km")
        self._conv_from.pack(side=tk.LEFT, padx=4)

        tk.Label(row1, text="To:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=6)
        self._conv_to = ttk.Combobox(row1, values=all_units, width=6,
                                      font=("Courier New", 10))
        self._conv_to.set("mi")
        self._conv_to.pack(side=tk.LEFT, padx=4)

        self._make_btn(row1, "Convert", self._do_unit_convert, "eq").pack(side=tk.LEFT, padx=12)

        self._conv_result = tk.Label(sec1, text="",
                                      bg=t["surface"], fg=t["accent2"],
                                      font=("Courier New", 12, "bold"))
        self._conv_result.pack(anchor="w", padx=12, pady=4)

        # ── Angle conversion ──
        sec2 = tk.LabelFrame(frame, text="  Angle Conversion  ",
                              bg=t["surface"], fg=t["accent"],
                              font=("Courier New", 10, "bold"),
                              bd=1, relief="groove")
        sec2.pack(fill=tk.X, padx=14, pady=6)

        row2 = tk.Frame(sec2, bg=t["surface"])
        row2.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(row2, text="Value:", bg=t["surface"],
                 fg=t["text_dim"], font=("Courier New", 9)).pack(side=tk.LEFT)
        self._ang_val = tk.Entry(row2, width=12, font=("Courier New", 12),
                                  bg=t["display_bg"], fg=t["display_fg"],
                                  insertbackground=t["accent"], relief="flat", bd=4)
        self._ang_val.insert(0, "180")
        self._ang_val.pack(side=tk.LEFT, padx=6)

        self._ang_from = ttk.Combobox(row2, values=["DEG", "RAD"], width=6,
                                       font=("Courier New", 10))
        self._ang_from.set("DEG")
        self._ang_from.pack(side=tk.LEFT, padx=4)

        tk.Label(row2, text="→", bg=t["surface"],
                 fg=t["text"], font=("Courier New", 12)).pack(side=tk.LEFT, padx=4)
        self._ang_to = ttk.Combobox(row2, values=["DEG", "RAD"], width=6,
                                     font=("Courier New", 10))
        self._ang_to.set("RAD")
        self._ang_to.pack(side=tk.LEFT, padx=4)

        self._make_btn(row2, "Convert", self._do_angle_convert, "eq").pack(side=tk.LEFT, padx=12)

        self._ang_result = tk.Label(sec2, text="",
                                     bg=t["surface"], fg=t["accent2"],
                                     font=("Courier New", 12, "bold"))
        self._ang_result.pack(anchor="w", padx=12, pady=4)

    def _do_unit_convert(self):
        try:
            val = float(self._conv_val.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number")
            return
        result = self.engine.convert_unit(val,
                                           self._conv_from.get(),
                                           self._conv_to.get())
        self._conv_result.config(text=result)
        self._refresh_history()

    def _do_angle_convert(self):
        try:
            val = float(self._ang_val.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number")
            return
        result = self.engine.angle_convert(val,
                                            self._ang_from.get(),
                                            self._ang_to.get())
        self._ang_result.config(text=result)
        self._refresh_history()

    # ── History Tab ────────────────────────────────────────
    def _build_history_tab(self):
        t = self.T
        frame = tk.Frame(self.notebook, bg=t["bg"])
        self.notebook.add(frame, text="  📋 History  ")

        ctrl = tk.Frame(frame, bg=t["surface"], pady=6)
        ctrl.pack(fill=tk.X, padx=10, pady=8)

        self._make_btn(ctrl, "⟳ Refresh",  self._refresh_history,  "fn").pack(side=tk.LEFT, padx=6)
        self._make_btn(ctrl, "🗑 Clear",    self._clear_history,    "clear").pack(side=tk.LEFT, padx=4)
        self._make_btn(ctrl, "💾 Save JSON", self._save_history_json, "special").pack(side=tk.LEFT, padx=4)
        self._make_btn(ctrl, "💾 Save TXT",  self._save_history_txt,  "special").pack(side=tk.LEFT, padx=4)

        self._history_box = scrolledtext.ScrolledText(
            frame, font=("Courier New", 10),
            bg=t["display_bg"], fg=t["text"],
            relief="flat", bd=4, state="disabled")
        self._history_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _refresh_history(self):
        self._history_box.config(state="normal")
        self._history_box.delete("1.0", tk.END)
        for entry in reversed(self.engine.history):
            line = (f"[{entry['timestamp']}]  "
                    f"({entry['angle_mode']})  "
                    f"{entry['expression']}  =  {entry['result']}\n")
            self._history_box.insert(tk.END, line)
        self._history_box.config(state="disabled")

    def _clear_history(self):
        self.engine.history.clear()
        self._refresh_history()

    def _save_history_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.engine.save_history(path)
            messagebox.showinfo("Saved", f"History saved to:\n{path}")

    def _save_history_txt(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            self.engine.export_history_txt(path)
            messagebox.showinfo("Saved", f"History saved to:\n{path}")

    # ── Helpers ────────────────────────────────────────────
    def _make_btn(self, parent, text, cmd, btype="num") -> tk.Button:
        t = self.T
        color_map = {
            "num":     (t["btn_num"],    t["text"]),
            "op":      (t["btn_op"],     t["accent"]),
            "fn":      (t["btn_fn"],     t["accent2"]),
            "eq":      (t["btn_eq"],     "#ffffff"),
            "clear":   (t["btn_clear"],  t["danger"]),
            "special": (t["btn_special"],t["warning"]),
        }
        bg, fg = color_map.get(btype, (t["btn_num"], t["text"]))
        return tk.Button(parent, text=text, command=cmd,
                          bg=bg, fg=fg,
                          font=("Courier New", 9, "bold"),
                          relief="flat", bd=0, cursor="hand2",
                          activebackground=t["accent"],
                          activeforeground="#ffffff",
                          padx=10, pady=5)

    def _toggle_theme(self):
        self.theme_name.set("light" if self.theme_name.get() == "dark" else "dark")
        # Rebuild UI
        for w in self.root.winfo_children():
            w.destroy()
        self._build_ui()


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("Advanced Scientific Calculator")

    # Centre window
    root.update_idletasks()
    w, h = 980, 720
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    app = ScientificCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
