import cmath
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np


class MicrowaveMatrixEngine:
  """Lõi tính toán giải tích ma trận mạng vi ba."""

  @staticmethod
  def y_to_s(Y: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    N = Y.shape[0]
    I = np.eye(N, dtype=complex)
    Y_bar = Y * Z0  # Y_bar = Y / Y0 = Y * Z0
    denom = I + Y_bar
    if np.abs(np.linalg.det(denom)) < 1e-12:
      raise ValueError("Ma trận (I + Y_norm) suy biến, không thể đảo ma trận!")
    return np.dot(I - Y_bar, np.linalg.inv(denom))

  @staticmethod
  def s_to_y(S: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    N = S.shape[0]
    I = np.eye(N, dtype=complex)
    denom = I + S
    if np.abs(np.linalg.det(denom)) < 1e-12:
      raise ValueError("Ma trận (I + S) suy biến, không thể đảo ma trận!")
    Y_bar = np.dot(I - S, np.linalg.inv(denom))
    return Y_bar / Z0

  @staticmethod
  def abcd_to_s(ABCD: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    A, B = ABCD[0, 0], ABCD[0, 1]
    C, D = ABCD[1, 0], ABCD[1, 1]
    delta = A + (B / Z0) + (C * Z0) + D
    if np.abs(delta) < 1e-12:
      raise ValueError("Mẫu số delta = A + B/Z0 + C*Z0 + D xấp xỉ 0!")

    det_abcd = A * D - B * C
    S11 = (A + (B / Z0) - (C * Z0) - D) / delta
    S12 = 2.0 * det_abcd / delta
    S21 = 2.0 / delta
    S22 = (-A + (B / Z0) - (C * Z0) + D) / delta
    return np.array([[S11, S12], [S21, S22]], dtype=complex)

  @staticmethod
  def s_to_abcd(S: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    S11, S12 = S[0, 0], S[0, 1]
    S21, S22 = S[1, 0], S[1, 1]
    if np.abs(S21) < 1e-12:
      raise ValueError(
          "Tham số S21 xấp xỉ 0 (mạng cách ly hoàn toàn), không tồn tại ma trận"
          " ABCD!"
      )

    delta_s = S11 * S22 - S12 * S21
    A = ((1.0 + S11) * (1.0 - S22) + S12 * S21) / (2.0 * S21)
    B = Z0 * (((1.0 + S11) * (1.0 + S22) - S12 * S21) / (2.0 * S21))
    C = (1.0 / Z0) * (((1.0 - S11) * (1.0 - S22) - S12 * S21) / (2.0 * S21))
    D = ((1.0 - S11) * (1.0 + S22) + S12 * S21) / (2.0 * S21)
    return np.array([[A, B], [C, D]], dtype=complex)

  @staticmethod
  def check_properties(S: np.ndarray, tol: float = 1e-5) -> dict:
    """Đánh giá tính tương hỗ, đối xứng và không tiêu tán qua ma trận S."""
    N = S.shape[0]
    # 1. Tương hỗ: S = S^T
    is_reciprocal = np.allclose(S, S.T, atol=tol)

    # 2. Đối xứng: S11 == S22 và S12 == S21 (đối với 2 cổng)
    is_symmetric = False
    if N == 2:
      is_symmetric = np.isclose(S[0, 0], S[1, 1], atol=tol) and np.isclose(
          S[0, 1], S[1, 0], atol=tol
      )
    else:
      diag_equal = all(np.isclose(S[i, i], S[0, 0], atol=tol) for i in range(N))
      is_symmetric = is_reciprocal and diag_equal

    # 3. Không tổn hao (Lossless): S^dag * S = I (Ma trận Unita)
    S_conj_T = np.conjugate(S.T)
    identity_check = np.dot(S_conj_T, S)
    is_lossless = np.allclose(identity_check, np.eye(N), atol=tol)

    return {
        "Reciprocal": is_reciprocal,
        "Symmetric": is_symmetric,
        "Lossless": is_lossless,
    }


class MicrowaveGUI:

  def __init__(self, root):
    self.root = root
    self.root.title("Phần mềm Chuyển đổi Ma trận Vi ba [Đề tài 3]")
    self.root.geometry("860x700")

    self.engine = MicrowaveMatrixEngine()
    self.entry_grid = []

    self._build_controls()
    self._rebuild_grid()

  def _build_controls(self):
    ctrl_frame = ttk.LabelFrame(
        self.root, text=" Cấu hình tham số ", padding=10
    )
    ctrl_frame.pack(fill="x", padx=10, pady=5)

    # Chế độ chuyển đổi
    ttk.Label(ctrl_frame, text="Phép chuyển đổi:").grid(
        row=0, column=0, sticky="w", padx=5
    )
    self.mode_var = tk.StringVar(value="Y -> S")
    mode_cb = ttk.Combobox(
        ctrl_frame,
        textvariable=self.mode_var,
        values=["Y -> S", "S -> Y", "ABCD -> S", "S -> ABCD"],
        state="readonly",
        width=15,
    )
    mode_cb.grid(row=0, column=1, padx=5, pady=5)
    mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

    # Số cổng N
    ttk.Label(ctrl_frame, text="Số cổng N:").grid(
        row=0, column=2, sticky="w", padx=5
    )
    self.n_var = tk.IntVar(value=2)
    self.n_spin = ttk.Spinbox(
        ctrl_frame,
        from_=2,
        to=6,
        textvariable=self.n_var,
        width=5,
        command=self._rebuild_grid,
    )
    self.n_spin.grid(row=0, column=3, padx=5, pady=5)

    # Trở kháng chuẩn Z0
    ttk.Label(ctrl_frame, text="Trở kháng Z0 (Ω):").grid(
        row=0, column=4, sticky="w", padx=5
    )
    self.z0_var = tk.DoubleVar(value=50.0)
    ttk.Entry(ctrl_frame, textvariable=self.z0_var, width=8).grid(
        row=0, column=5, padx=5, pady=5
    )

    # Nút bấm hành động
    btn_frame = ttk.Frame(self.root, padding=5)
    btn_frame.pack(fill="x", padx=10)
    ttk.Button(
        btn_frame, text="Thực hiện Chuyển đổi", command=self.process_conversion
    ).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Tải Dữ liệu Mẫu", command=self.load_sample).pack(
        side="left", padx=5
    )
    ttk.Button(btn_frame, text="Xóa Dữ liệu", command=self.clear_grid).pack(
        side="left", padx=5
    )

    # Khung ma trận đầu vào
    self.grid_frame = ttk.LabelFrame(
        self.root, text=" Nhập Ma trận Nguồn (định dạng: a+bj) ", padding=10
    )
    self.grid_frame.pack(fill="x", padx=10, pady=5)

    # Khung kết quả
    res_frame = ttk.LabelFrame(self.root, text=" Ma trận Kết quả ", padding=10)
    res_frame.pack(fill="both", expand=True, padx=10, pady=5)

    self.res_text = tk.Text(
        res_frame, height=12, font=("Courier New", 10), wrap="none"
    )
    self.res_text.pack(fill="both", expand=True)

    # Panel tính chất mạng
    self.prop_label = ttk.Label(
        self.root,
        text="Tính chất mạng: Chưa xác định",
        font=("Arial", 10, "bold"),
        foreground="#1E3A8A",
    )
    self.prop_label.pack(anchor="w", padx=15, pady=5)

  def _on_mode_change(self, event=None):
    mode = self.mode_var.get()
    if "ABCD" in mode:
      self.n_var.set(2)
      self.n_spin.config(state="disabled")
    else:
      self.n_spin.config(state="normal")
    self._rebuild_grid()

  def _rebuild_grid(self):
    for row in self.entry_grid:
      for e in row:
        e.destroy()
    self.entry_grid.clear()

    N = self.n_var.get()
    mode = self.mode_var.get()
    prefix = mode.split(" ")[0]

    for i in range(N):
      row_entries = []
      for j in range(N):
        lbl = ttk.Label(
            self.grid_frame,
            text=f"{prefix}{i+1}{j+1}:",
            font=("Arial", 8, "italic"),
        )
        lbl.grid(row=i * 2, column=j, padx=2, pady=1)
        entry = ttk.Entry(self.grid_frame, width=14, justify="center")
        entry.insert(0, "0")
        entry.grid(row=i * 2 + 1, column=j, padx=4, pady=2)
        row_entries.append(entry)
      self.entry_grid.append(row_entries)

  def _parse_input_matrix(self) -> np.ndarray:
    N = self.n_var.get()
    mat = np.zeros((N, N), dtype=complex)
    for i in range(N):
      for j in range(N):
        txt = self.entry_grid[i][j].get().strip().replace(" ", "")
        try:
          mat[i, j] = complex(txt)
        except ValueError:
          raise ValueError(
              f"Phần tử hàng {i+1}, cột {j+1} không đúng định dạng số phức"
              " (VD: 0.02, 1-2j)!"
          )
    return mat

  def process_conversion(self):
    try:
      Z0 = self.z0_var.get()
      if Z0 <= 0:
        messagebox.showerror("Lỗi", "Trở kháng Z0 phải là số thực dương!")
        return

      in_mat = self._parse_input_matrix()
      mode = self.mode_var.get()

      if mode == "Y -> S":
        out_mat = self.engine.y_to_s(in_mat, Z0)
        s_eval = out_mat
        title_out = "MA TRẬN TÁN XẠ [S]:"
      elif mode == "S -> Y":
        out_mat = self.engine.s_to_y(in_mat, Z0)
        s_eval = in_mat
        title_out = "MA TRẬN DẪN NẠP [Y] (Siemens):"
      elif mode == "ABCD -> S":
        out_mat = self.engine.abcd_to_s(in_mat, Z0)
        s_eval = out_mat
        title_out = "MA TRẬN TÁN XẠ [S]:"
      elif mode == "S -> ABCD":
        out_mat = self.engine.s_to_abcd(in_mat, Z0)
        s_eval = in_mat
        title_out = "MA TRẬN CASCADE [ABCD]:"

      props = self.engine.check_properties(s_eval)
      self._display_results(out_mat, title_out, props)

    except Exception as e:
      messagebox.showerror("Lỗi xử lý", str(e))

  def _display_results(
      self, mat: np.ndarray, title: str, props: dict[str, bool]
  ):
    self.res_text.delete("1.0", tk.END)
    self.res_text.insert(tk.END, f"{'='*60}\n {title}\n{'='*60}\n\n")

    N = mat.shape[0]
    self.res_text.insert(
        tk.END, "--- 1. Biểu diễn Cartesian (Phần Thực + j*Phần Ảo) ---\n"
    )
    for i in range(N):
      row_str = "  ".join([f"{mat[i, j]:.6f}" for j in range(N)])
      self.res_text.insert(tk.END, f"  [ {row_str} ]\n")

    self.res_text.insert(
        tk.END, "\n--- 2. Biểu diễn Cực (Biên độ ∠ Góc Pha độ) ---\n"
    )
    for i in range(N):
      row_str = "  ".join([
          f"{abs(mat[i, j]):.5f} ∠ {np.degrees(cmath.phase(mat[i, j])):.2f}°"
          for j in range(N)
      ])
      self.res_text.insert(tk.END, f"  [ {row_str} ]\n")

    # Hiển thị tính chất
    prop_str = (
        f"Tính chất mạng vi ba: "
        f"Tương hỗ (Reciprocal): {'CÓ' if props['Reciprocal'] else 'KHÔNG'}  |  "
        f"Đối xứng (Symmetrical): {'CÓ' if props['Symmetric'] else 'KHÔNG'}  |  "
        f"Không tiêu tán (Lossless): {'CÓ' if props['Lossless'] else 'KHÔNG'}"
    )
    self.prop_label.config(text=prop_str)

  def load_sample(self):
    """Tải dữ liệu mẫu: Đường dây truyền sóng 1/4 bước sóng (Z0 = 50 Ohm)."""
    self.mode_var.set("ABCD -> S")
    self._on_mode_change()
    # A = 0, B = 50j, C = 0.02j, D = 0
    sample = [["0", "50j"], ["0.02j", "0"]]
    for i in range(2):
      for j in range(2):
        self.entry_grid[i][j].delete(0, tk.END)
        self.entry_grid[i][j].insert(0, sample[i][j])

  def clear_grid(self):
    N = self.n_var.get()
    for i in range(N):
      for j in range(N):
        self.entry_grid[i][j].delete(0, tk.END)
        self.entry_grid[i][j].insert(0, "0")
    self.res_text.delete("1.0", tk.END)
    self.prop_label.config(text="Tính chất mạng: Chưa xác định")


if __name__ == "__main__":
  root = tk.Tk()
  app = MicrowaveGUI(root)
  root.mainloop()
