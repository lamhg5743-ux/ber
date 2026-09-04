import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Dự đoán BER với GPR | Kênh AWGN",
    layout="wide"
)

# ---------------------------------------------------------
# 1. HÀM TÍNH TOÁN & SINH DỮ LIỆU HUẤN LUYỆN
# ---------------------------------------------------------
def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2))

def get_theoretical_ber(ebno_db, mod_type, code_rate):
    """Tính BER lý thuyết (xấp xỉ có xét đến Code Rate)"""
    ebno_lin = 10 ** (ebno_db / 10.0)
    effective_ebno = ebno_lin * code_rate

    if mod_type == "BPSK":
        ber = q_func(np.sqrt(2 * effective_ebno))
    elif mod_type == "QPSK":
        ber = q_func(np.sqrt(2 * effective_ebno))
    elif mod_type == "8-PSK":
        ber = (2 / 3) * q_func(np.sqrt(6 * effective_ebno) * np.sin(np.pi / 8))
    elif mod_type == "16-QAM":
        ber = (3 / 4) * q_func(np.sqrt((4 / 5) * effective_ebno))
    elif mod_type == "64-QAM":
        ber = (7 / 12) * q_func(np.sqrt((2 / 7) * effective_ebno))
    else:
        ber = 1e-1
    return np.clip(ber, 1e-7, 0.5)

@st.cache_resource
def train_gpr_model():
    """Tạo tập dữ liệu, chia tập Train/Test, huấn luyện và đánh giá GPR"""
    mod_mapping = {"BPSK": 1, "QPSK": 2, "8-PSK": 3, "16-QAM": 4, "64-QAM": 6}
    code_rates = [1/2, 2/3, 3/4, 5/6, 1.0]
    ebno_range = np.linspace(-2, 14, 21)

    X_all = []
    y_all = []

    np.random.seed(42)
    for mod_name, mod_bits in mod_mapping.items():
        for cr in code_rates:
            for ebno in ebno_range:
                ber = get_theoretical_ber(ebno, mod_name, cr)
                log_ber = np.log10(ber)
                noisy_log_ber = log_ber + np.random.normal(0, 0.03)
                
                X_all.append([ebno, mod_bits, cr])
                y_all.append(noisy_log_ber)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Chia tập dữ liệu: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Thiết lập Kernel cho GPR
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[2.0, 1.0, 0.5]) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X_train, y_train)

    # Tính toán các chỉ số độ chính xác trên tập Test
    y_pred_test, std_test = gpr.predict(X_test, return_std=True)
    mae_val = mean_absolute_error(y_test, y_pred_test)
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_val = r2_score(y_test, y_pred_test)

    metrics = {
        "MAE": mae_val,
        "RMSE": rmse_val,
        "R2": r2_val,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test
    }

    return gpr, mod_mapping, metrics

# ---------------------------------------------------------
# 2. GIAO DIỆN & TƯƠNG TÁC NGƯỜI DÙNG
# ---------------------------------------------------------
st.title("Dự đoán BER của các sơ đồ điều chế trên kênh AWGN")
st.caption("Mô hình hồi quy phi tuyến tính sử dụng **Gaussian Process Regression (GPR)**")

# Sidebar điều khiển
st.sidebar.header("Tham số đầu vào")

mod_selected = st.sidebar.selectbox(
    "Sơ đồ điều chế:",
    ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"],
    index=1
)

code_rate_label = st.sidebar.selectbox(
    "Tốc độ mã (Code Rate):",
    ["1/2", "2/3", "3/4", "5/6", "1 (Không mã hóa)"],
    index=4
)
code_rate_dict = {"1/2": 0.5, "2/3": 2/3, "3/4": 0.75, "5/6": 5/6, "1 (Không mã hóa)": 1.0}
cr_selected = code_rate_dict[code_rate_label]

ebno_input = st.sidebar.slider(
    "Tỷ số $E_b/N_0$ dự đoán (dB):",
    min_value=-2.0,
    max_value=14.0,
    value=6.0,
    step=0.5
)

# Load model & metrics
gpr_model, mod_map, eval_metrics = train_gpr_model()

# Dự đoán điểm người dùng chọn
mod_bit_val = mod_map[mod_selected]
query_point = np.array([[ebno_input, mod_bit_val, cr_selected]])
pred_log_ber, sigma = gpr_model.predict(query_point, return_std=True)

pred_ber = 10 ** (pred_log_ber[0])
ber_lower = 10 ** (pred_log_ber[0] - 1.96 * sigma[0])
ber_upper = 10 ** (pred_log_ber[0] + 1.96 * sigma[0])
theo_ber = get_theoretical_ber(ebno_input, mod_selected, cr_selected)

# ---------------------------------------------------------
# 3. HIỂN THỊ KẾT QUẢ DỰ ĐOÁN & CHỈ SỐ ĐỘ CHÍNH XÁC
# ---------------------------------------------------------
st.subheader("🎯 Kết quả Dự đoán & Chỉ số Đánh giá Độ chính xác")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label=f"BER Dự đoán ({ebno_input} dB)",
        value=f"{pred_ber:.2e}"
    )

with col2:
    st.metric(
        label="BER Lý thuyết",
        value=f"{theo_ber:.2e}"
    )

with col3:
    st.metric(
        label="Hệ số $R^2$ (Độ khớp)",
        value=f"{eval_metrics['R2']:.4f}"
    )

with col4:
    st.metric(
        label="Sai số MAE",
        value=f"{eval_metrics['MAE']:.4f}"
    )

with col5:
    st.metric(
        label="Sai số RMSE",
        value=f"{eval_metrics['RMSE']:.4f}"
    )

st.markdown("---")

# ---------------------------------------------------------
# 4. VẼ ĐỒ THỊ BER ĐƯỜNG CONG VÀ KHOẢNG TIN CẬY
# ---------------------------------------------------------
st.subheader(f"Đường cong đặc tuyến BER - Điều chế {mod_selected} (Tốc độ mã: {code_rate_label})")

ebno_dense = np.linspace(-2, 14, 100)
X_test_curve = np.array([[e, mod_bit_val, cr_selected] for e in ebno_dense])
y_pred_curve, y_std_curve = gpr_model.predict(X_test_curve, return_std=True)

ber_pred_curve = 10 ** y_pred_curve
ber_pred_upper = 10 ** (y_pred_curve + 1.96 * y_std_curve)
ber_pred_lower = 10 ** (y_pred_curve - 1.96 * y_std_curve)
ber_theo_curve = [get_theoretical_ber(e, mod_selected, cr_selected) for e in ebno_dense]

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.semilogy(ebno_dense, ber_theo_curve, 'k--', label="BER Lý thuyết", linewidth=1.5)
ax.semilogy(ebno_dense, ber_pred_curve, 'b-', label="BER Dự đoán (GPR)", linewidth=2)
ax.fill_between(
    ebno_dense,
    ber_pred_lower,
    ber_pred_upper,
    color='blue',
    alpha=0.18,
    label="Khoảng tin cậy 95% ($\pm 1.96\sigma$)"
)
ax.scatter([ebno_input], [pred_ber], color='red', s=100, zorder=5, label=f"Điểm chọn ({ebno_input} dB)")

ax.set_title(f"Hiệu năng BER trên kênh AWGN ({mod_selected})", fontsize=13)
ax.set_xlabel("$E_b/N_0$ (dB)", fontsize=11)
ax.set_ylabel("Tỷ lệ lỗi bit (BER)", fontsize=11)
ax.set_ylim([1e-6, 1])
ax.set_xlim([-2, 14])
ax.grid(True, which="both", ls=":", alpha=0.6)
ax.legend(loc="upper right")

st.pyplot(fig)

# ---------------------------------------------------------
# 5. BẢNG KIỂM TRA ĐỘ CHÍNH XÁC (MẪU TẬP TEST)
# ---------------------------------------------------------
st.subheader("📋 Bảng Kiểm tra Độ chính xác (So sánh BER Mô phỏng vs BER Dự đoán)")

rev_map = {1: "BPSK", 2: "QPSK", 3: "8-PSK", 4: "16-QAM", 6: "64-QAM"}
test_samples_table = []

# Lấy 6 mẫu ngẫu nhiên từ tập kiểm thử
for i in range(min(6, len(eval_metrics["y_test"]))):
    x_i = eval_metrics["X_test"][i]
    y_true_log = eval_metrics["y_test"][i]
    y_pred_log = eval_metrics["y_pred_test"][i]
    
    val_true = 10 ** y_true_log
    val_pred = 10 ** y_pred_log
    
    test_samples_table.append({
        "Sơ đồ điều chế": rev_map.get(int(x_i[1]), "Unknown"),
        "Eb/N0 (dB)": f"{x_i[0]:.1f}",
        "Code Rate": f"{x_i[2]:.2f}",
        "BER Mô phỏng (Thực tế)": f"{val_true:.3e}",
        "BER Dự đoán (GPR)": f"{val_pred:.3e}",
        "Sai số tuyệt đối": f"{abs(val_true - val_pred):.3e}"
    })

st.dataframe(test_samples_table)

# ---------------------------------------------------------
# 6. BẢNG SO SÁNH TẤT CẢ CÁC SƠ ĐỒ ĐIỀU CHẾ
# ---------------------------------------------------------
st.subheader(f"📊 So sánh tất cả các sơ đồ điều chế tại $E_b/N_0 = {ebno_input}$ dB")

all_mods = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"]
comparison_data = []

for m in all_mods:
    m_val = mod_map[m]
    q_pt = np.array([[ebno_input, m_val, cr_selected]])
    log_p, s = gpr_model.predict(q_pt, return_std=True)
    p_val = 10 ** log_p[0]
    comparison_data.append({
        "Sơ đồ điều chế": m,
        "Số bit / Symbol": m_val,
        "Code Rate": code_rate_label,
        "BER Dự đoán (GPR)": f"{p_val:.3e}",
        "Độ bất định (Std Dev)": f"{s[0]:.4f}"
    })

st.dataframe(comparison_data)