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
    page_title="Mô phỏng & Dự đoán BER bằng GPR", 
    page_icon="📡", 
    layout="wide"
)

# 1. Hàm tính BER lý thuyết
def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

def get_theoretical_ber(ebno_db, mod_name, code_rate):
    ebno_lin = 10.0 ** (ebno_db / 10.0)
    eff_ebno = ebno_lin * code_rate
    
    if mod_name in ["BPSK", "QPSK"]:
        ber = q_func(np.sqrt(2.0 * eff_ebno))
    elif mod_name == "8-PSK":
        ber = (2.0 / 3.0) * q_func(np.sqrt(6.0 * eff_ebno) * np.sin(np.pi / 8.0))
    elif mod_name == "16-QAM":
        ber = 0.75 * q_func(np.sqrt(0.8 * eff_ebno))
    elif mod_name == "64-QAM":
        ber = (7.0 / 12.0) * q_func(np.sqrt((2.0 / 7.0) * eff_ebno))
    else:
        ber = q_func(np.sqrt(2.0 * eff_ebno))
    return np.clip(ber, 1e-7, 0.5)

# 2. Huấn luyện mô hình Gaussian Process Regression (GPR)
@st.cache_resource
def train_gpr_model():
    np.random.seed(42)
    mod_mapping = {"BPSK": 1, "QPSK": 2, "8-PSK": 3, "16-QAM": 4, "64-QAM": 6}
    code_rates = [0.5, 2/3, 0.75, 5/6, 1.0]
    ebno_range = np.linspace(-2, 14, 21)

    X_list, y_list = [], []
    for mod_name, bits in mod_mapping.items():
        for cr in code_rates:
            for ebno in ebno_range:
                ber_theo = get_theoretical_ber(ebno, mod_name, cr)
                log_ber = np.log10(ber_theo)
                noisy_log_ber = log_ber + np.random.normal(0, 0.03)
                X_list.append([ebno, bits, cr])
                y_list.append(noisy_log_ber)

    X = np.array(X_list)
    y = np.array(y_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[3.0, 2.0, 0.5], length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X_train, y_train)

    y_pred_test = gpr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    return gpr, mae, rmse, r2

# Nạp mô hình đã huấn luyện
gpr_model, mae, rmse, r2 = train_gpr_model()

# 3. Thanh công cụ bên trái (Sidebar)
st.sidebar.header("⚙️ Thiết lập Tham số")

mod_options = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"]
mod_selected = st.sidebar.selectbox("Sơ đồ điều chế", mod_options, index=0)

cr_dict = {
    "1/2": 0.5,
    "2/3": 2/3,
    "3/4": 0.75,
    "5/6": 5/6,
    "1.0 (Không mã hóa)": 1.0
}
cr_label = st.sidebar.selectbox("Tốc độ mã kênh (Rc)", list(cr_dict.keys()), index=0)
cr_selected = cr_dict[cr_label]

ebno_input = st.sidebar.slider(
    "Tỷ số Eb/N0 (dB)", 
    min_value=-2.0, 
    max_value=14.0, 
    value=4.0, 
    step=0.5
)

# Mã QR truy cập
st.sidebar.markdown("---")
st.sidebar.subheader("📱 Quét mã trải nghiệm")
app_url = "https://share.streamlit.io"
qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={app_url}"
st.sidebar.image(qr_url, caption="Quét bằng camera điện thoại")

mod_bits = {"BPSK": 1, "QPSK": 2, "8-PSK": 3, "16-QAM": 4, "64-QAM": 6}
bit_val = mod_bits[mod_selected]

# 4. Khu vực nội dung chính
st.title("📡 Dự đoán Hiệu năng BER trên Kênh AWGN bằng Mô hình GPR")

# Khối hiển thị độ đo đánh giá mô hình
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("MAE (log10)", f"{mae:.4f}")
col_m2.metric("RMSE (log10)", f"{rmse:.4f}")
col_m3.metric("R² Score", f"{r2:.4f}")
col_m4.metric("Thông lượng hữu ích", f"{bit_val * cr_selected:.2f} bits/kênh")

# Dự đoán giá trị tại điểm người dùng chọn
point_query = np.array([[ebno_input, bit_val, cr_selected]])
log_ber_pred, pred_std = gpr_model.predict(point_query, return_std=True)
ber_pred = 10.0 ** log_ber_pred[0]
ber_theo = get_theoretical_ber(ebno_input, mod_selected, cr_selected)

# 5. Vẽ đồ thị hiệu năng BER
ebno_curve = np.linspace(-2, 14, 100)
theo_curve = [get_theoretical_ber(e, mod_selected, cr_selected) for e in ebno_curve]

X_curve = np.array([[e, bit_val, cr_selected] for e in ebno_curve])
log_pred_curve, std_curve = gpr_model.predict(X_curve, return_std=True)

pred_curve = 10.0 ** log_pred_curve
lower_ci = 10.0 ** (log_pred_curve - 1.96 * std_curve)
upper_ci = 10.0 ** (log_pred_curve + 1.96 * std_curve)

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(ebno_curve, theo_curve, "k--", linewidth=1.5, label="BER Lý thuyết")
ax.plot(ebno_curve, pred_curve, "b-", linewidth=2.0, label="BER Dự đoán (GPR)")
ax.fill_between(ebno_curve, lower_ci, upper_ci, color="blue", alpha=0.18, label="Khoảng tin cậy 95% (±1.96σ)")
ax.plot(ebno_input, ber_pred, "ro", markersize=10, label=f"Điểm chọn ({ebno_input:.1f} dB)")

ax.set_yscale("log")
ax.set_xlim(-2, 14)
ax.set_ylim(1e-6, 1.0)
ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
ax.set_ylabel("Tỷ lệ lỗi bit (BER)", fontsize=12)
ax.set_title(f"Hiệu năng BER trên kênh AWGN ({mod_selected})", fontsize=14)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True)

st.pyplot(fig)

# Thống kê chi tiết tại điểm đang xét
st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.info(f"**BER Lý thuyết:** {ber_theo:.4e}")
c2.success(f"**BER Dự đoán (GPR):** {ber_pred:.4e}")
c3.warning(f"**Sai số tuyệt đối:** {abs(ber_theo - ber_pred):.4e}")
