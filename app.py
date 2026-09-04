import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình trang web Streamlit
st.set_page_config(
    page_title="Dự đoán BER bằng GPR",
    page_icon="📡",
    layout="wide"
)

# 1. Hàm tính BER lý thuyết chuẩn trên kênh AWGN
def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

def get_theoretical_ber(ebno_db, mod_name, code_rate):
    ebno_lin = 10.0 ** (ebno_db / 10.0)
    ebno_eff = ebno_lin * code_rate
    
    if mod_name == 'BPSK':
        ber = q_func(np.sqrt(2.0 * ebno_eff))
    elif mod_name == 'QPSK':
        ber = q_func(np.sqrt(2.0 * ebno_eff))
    elif mod_name == '8-PSK':
        ber = (2.0 / 3.0) * q_func(np.sqrt(6.0 * ebno_eff) * np.sin(np.pi / 8.0))
    elif mod_name == '16-QAM':
        ber = 0.75 * q_func(np.sqrt(0.8 * ebno_eff))
    elif mod_name == '64-QAM':
        ber = (7.0 / 12.0) * q_func(np.sqrt((2.0 / 7.0) * ebno_eff))
    else:
        ber = q_func(np.sqrt(2.0 * ebno_eff))
        
    return np.clip(ber, 1e-7, 0.5)

# 2. Tạo dữ liệu và huấn luyện mô hình Gaussian Process Regression (GPR)
@st.cache_resource
def train_gpr_model():
    np.random.seed(42)
    mod_mapping = {
        'BPSK': 1,
        'QPSK': 2,
        '8-PSK': 3,
        '16-QAM': 4,
        '64-QAM': 6
    }
    code_rates = [0.5, 0.67, 0.75, 0.83, 1.0]
    ebno_range = np.linspace(-2.0, 14.0, 21)
    
    X_list = []
    y_list = []
    
    for mod_name, bit_val in mod_mapping.items():
        for cr in code_rates:
            for ebno in ebno_range:
                ber_theo = get_theoretical_ber(ebno, mod_name, cr)
                log_ber = np.log10(ber_theo)
                # Mô phỏng sai số thực nghiệm qua nhiễu Gauss
                log_ber_noisy = log_ber + np.random.normal(0.0, 0.03)
                
                X_list.append([ebno, bit_val, cr])
                y_list.append(log_ber_noisy)
                
    X = np.array(X_list)
    y = np.array(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[2.0, 1.0, 0.5], length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X_train, y_train)
    
    y_pred_test = gpr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    return gpr, mae, rmse, r2, mod_mapping

# Tải mô hình
gpr_model, mae_score, rmse_score, r2_val, mod_mapping = train_gpr_model()

# 3. Giao diện người dùng trên thanh Sidebar
st.sidebar.title("⚙️ Tham số hệ thống")

mod_selected = st.sidebar.selectbox(
    "Sơ đồ điều chế:",
    ['BPSK', 'QPSK', '8-PSK', '16-QAM', '64-QAM'],
    index=0
)

code_rate_dict = {
    '1/2': 0.5,
    '2/3': 0.67,
    '3/4': 0.75,
    '5/6': 0.83,
    '1 (Không mã hóa)': 1.0
}
code_rate_label = st.sidebar.selectbox(
    "Tốc độ mã hóa kênh (Code Rate):",
    list(code_rate_dict.keys()),
    index=0
)
cr_selected = code_rate_dict[code_rate_label]

ebno_input = st.sidebar.slider(
    "Tỷ số Eb/N0 (dB):",
    min_value=-2.0,
    max_value=14.0,
    value=4.0,
    step=0.5
)

# Mã QR chia sẻ ứng dụng trên Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📱 Quét mã trải nghiệm")
app_url = "https://share.streamlit.io"
qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={app_url}"
st.sidebar.image(qr_url, caption="Mở trên điện thoại")

# 4. Giao diện nội dung chính
st.title("📡 Mô phỏng & Dự đoán Hiệu năng BER bằng GPR")
st.markdown(
    "Ứng dụng trí tuệ nhân tạo **Gaussian Process Regression (GPR)** "
    "để dự đoán tỷ lệ lỗi bit (BER) và ước lượng độ bất định trên kênh truyền vô tuyến AWGN."
)

# Hiển thị các chỉ số đánh giá tổng thể của mô hình AI
col1, col2, col3, col4 = st.columns(4)
col1.metric("Hệ số xác định (R²)", f"{r2_val:.4f}")
col2.metric("Sai số tuyệt đối (MAE)", f"{mae_score:.4f}")
col3.metric("Sai số bình phương (RMSE)", f"{rmse_score:.4f}")

mod_bit_val = mod_mapping[mod_selected]
query_point = np.array([[ebno_input, mod_bit_val, cr_selected]])
pred_log_ber, pred_std = gpr_model.predict(query_point, return_std=True)

pred_ber_val = 10.0 ** pred_log_ber[0]
theo_ber_val = get_theoretical_ber(ebno_input, mod_selected, cr_selected)

col4.metric(
    label=f"BER tại {ebno_input:.1f} dB",
    value=f"{pred_ber_val:.3e}",
    delta=f"Lý thuyết: {theo_ber_val:.3e}",
    delta_color="off"
)

st.markdown("---")

# 5. Vẽ đồ thị so sánh BER và khoảng tin cậy 95%
ebno_curve = np.linspace(-2.0, 14.0, 100)
X_curve = np.array([[e, mod_bit_val, cr_selected] for e in ebno_curve])

y_pred_curve_log, y_std_curve = gpr_model.predict(X_curve, return_std=True)

ber_pred_curve = 10.0 ** y_pred_curve_log
ber_lower_curve = 10.0 ** (y_pred_curve_log - 1.96 * y_std_curve)
ber_upper_curve = 10.0 ** (y_pred_curve_log + 1.96 * y_std_curve)

ber_theo_curve = [get_theoretical_ber(e, mod_selected, cr_selected) for e in ebno_curve]

fig, ax = plt.subplots(figsize=(10, 5))

# Đường lý thuyết
ax.plot(
    ebno_curve, 
    ber_theo_curve, 
    'k--', 
    label='BER Lý thuyết', 
    linewidth=1.8
)

# Đường dự đoán của GPR
ax.plot(
    ebno_curve, 
    ber_pred_curve, 
    'b-', 
    label='BER Dự đoán (GPR)', 
    linewidth=2.2
)

# Dải tin cậy 95%
ax.fill_between(
    ebno_curve, 
    ber_lower_curve, 
    ber_upper_curve, 
    color='blue', 
    alpha=0.2, 
    label='Khoảng tin cậy 95% (±1.96σ)'
)

# Điểm truy vấn hiện tại được chọn
ax.plot(
    ebno_input, 
    pred_ber_val, 
    'ro', 
    markersize=10, 
    label=f'Điểm chọn ({ebno_input:.1f} dB)'
)

ax.set_yscale('log')
ax.set_xlim(-2.0, 14.0)
ax.set_ylim(1e-6, 1.0)
ax.set_xlabel('$E_b/N_0$ (dB)', fontsize=12)
ax.set_ylabel('Tỷ lệ lỗi bit (BER)', fontsize=12)
ax.set_title(f'Hiệu năng BER trên kênh AWGN ({mod_selected})', fontsize=14)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc='upper right', fontsize=11)

st.pyplot(fig)
