import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Nghiên Cứu Hiệu Năng BER & Hệ Thống AMC Dùng GPR",
    page_icon="📡",
    layout="wide"
)

# 2. Các hàm giải tích tính BER lý thuyết
def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

def get_theoretical_ber(ebno_db, mod_name, code_rate):
    ebno_lin = 10.0 ** (ebno_db / 10.0)
    ebno_eff = ebno_lin * code_rate
    
    if mod_name in ['BPSK', 'QPSK']:
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

# Danh mục cấu hình MCS chuẩn cho hệ thống AMC
MCS_PROFILES = [
    {"name": "MCS 1: BPSK (R=1/2)", "mod": "BPSK", "bits": 1, "cr": 0.50, "raw_tp": 0.50},
    {"name": "MCS 2: QPSK (R=1/2)", "mod": "QPSK", "bits": 2, "cr": 0.50, "raw_tp": 1.00},
    {"name": "MCS 3: QPSK (R=3/4)", "mod": "QPSK", "bits": 2, "cr": 0.75, "raw_tp": 1.50},
    {"name": "MCS 4: 16-QAM (R=1/2)", "mod": "16-QAM", "bits": 4, "cr": 0.50, "raw_tp": 2.00},
    {"name": "MCS 5: 16-QAM (R=3/4)", "mod": "16-QAM", "bits": 4, "cr": 0.75, "raw_tp": 3.00},
    {"name": "MCS 6: 64-QAM (R=3/4)", "mod": "64-QAM", "bits": 6, "cr": 0.75, "raw_tp": 4.50},
    {"name": "MCS 7: 64-QAM (R=5/6)", "mod": "64-QAM", "bits": 6, "cr": 0.83, "raw_tp": 5.00},
]

# 3. Huấn luyện mô hình GPR (chạy 1 lần duy nhất và lưu vào cache)
@st.cache_resource
def train_gpr_model():
    np.random.seed(42)
    mod_mapping = {'BPSK': 1, 'QPSK': 2, '8-PSK': 3, '16-QAM': 4, '64-QAM': 6}
    code_rates = [0.5, 0.67, 0.75, 0.83, 1.0]
    ebno_range = np.linspace(-2.0, 20.0, 25)
    
    X_list, y_list = [], []
    for mod_name, bit_val in mod_mapping.items():
        for cr in code_rates:
            for ebno in ebno_range:
                ber_theo = get_theoretical_ber(ebno, mod_name, cr)
                log_ber_noisy = np.log10(ber_theo) + np.random.normal(0.0, 0.025)
                X_list.append([ebno, bit_val, cr])
                y_list.append(log_ber_noisy)
                
    X = np.array(X_list)
    y = np.array(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[2.0, 1.0, 0.5]) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
    gpr.fit(X_train, y_train)
    
    y_pred_test = gpr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    return gpr, mae, rmse, r2, mod_mapping

gpr_model, mae_val, rmse_val, r2_val, mod_mapping = train_gpr_model()

# 4. Thiết lập thanh Sidebar chung
st.sidebar.title("🎛️ Bảng Điều Khiển")
ebno_slider = st.sidebar.slider(
    "Tỷ số Eb/N0 của kênh (dB):",
    min_value=-2.0, max_value=20.0, value=6.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.subheader("📱 Quét mã trải nghiệm")
app_url = "https://share.streamlit.io"
qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=160x160&data={app_url}"
st.sidebar.image(qr_url, caption="Mở trên điện thoại")

# 5. Phân chia 2 Tab chức năng trên giao diện chính
tab1, tab2 = st.tabs([
    "📊 TAB 1: Khảo Sát & Đánh Giá Mô Hình BER (GPR)", 
    "📶 TAB 2: Hệ Thống Thích Ứng Kênh Truyền (AMC) Tự Động"
])

# ==========================================
# NỘI DUNG TAB 1: BÀI CŨ (DỰ ĐOÁN ĐƯỜNG CONG BER)
# ==========================================
with tab1:
    st.header("Khảo Sát Đường Đặc Tuyến BER Với Từng Sơ Đồ Điều Chế")
    
    col_t1_1, col_t1_2 = st.columns(2)
    with col_t1_1:
        mod_selected = st.selectbox(
            "Chọn sơ đồ điều chế cần khảo sát:",
            ['BPSK', 'QPSK', '8-PSK', '16-QAM', '64-QAM'],
            index=3
        )
    with col_t1_2:
        code_rate_dict = {'1/2': 0.5, '2/3': 0.67, '3/4': 0.75, '5/6': 0.83, '1 (Không mã hóa)': 1.0}
        cr_label = st.selectbox("Tốc độ mã hóa kênh (Code Rate):", list(code_rate_dict.keys()), index=0)
        cr_selected = code_rate_dict[cr_label]
        
    # Metrics độ chính xác
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hệ số xác định (R²)", f"{r2_val:.4f}")
    c1.caption("Độ tin cậy của thuật toán GPR")
    c2.metric("Sai số MAE", f"{mae_val:.4f}")
    c3.metric("Sai số RMSE", f"{rmse_val:.4f}")
    
    # Dự đoán tại điểm Eb/N0 hiện tại
    bit_val = mod_mapping[mod_selected]
    q_pt = np.array([[ebno_slider, bit_val, cr_selected]])
    log_pred, std_pred = gpr_model.predict(q_pt, return_std=True)
    ber_pred_val = 10.0 ** log_pred[0]
    ber_theo_val = get_theoretical_ber(ebno_slider, mod_selected, cr_selected)
    
    c4.metric(
        label=f"BER tại {ebno_slider:.1f} dB",
        value=f"{ber_pred_val:.3e}",
        delta=f"Lý thuyết: {ber_theo_val:.3e}",
        delta_color="off"
    )
    
    # Vẽ đồ thị BER kèm khoảng tin cậy 95%
    ebno_axis = np.linspace(-2.0, 16.0, 100)
    X_eval = np.array([[e, bit_val, cr_selected] for e in ebno_axis])
    y_log_eval, y_std_eval = gpr_model.predict(X_eval, return_std=True)
    
    ber_curve = 10.0 ** y_log_eval
    ber_lower = 10.0 ** (y_log_eval - 1.96 * y_std_eval)
    ber_upper = 10.0 ** (y_log_eval + 1.96 * y_std_eval)
    ber_theo_axis = [get_theoretical_ber(e, mod_selected, cr_selected) for e in ebno_axis]
    
    fig1, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ebno_axis, ber_theo_axis, 'k--', label='Lý thuyết AWGN', linewidth=1.8)
    ax.plot(ebno_axis, ber_curve, 'b-', label='GPR Dự đoán', linewidth=2.2)
    ax.fill_between(ebno_axis, ber_lower, ber_upper, color='blue', alpha=0.18, label='Khoảng tin cậy 95% (±1.96σ)')
    ax.plot(ebno_slider, ber_pred_val, 'ro', markersize=9, label=f'Điểm chọn ({ebno_slider:.1f} dB)')
    ax.set_yscale('log')
    ax.set_xlim(-2.0, 16.0)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel('$E_b/N_0$ (dB)', fontsize=11)
    ax.set_ylabel('Tỷ lệ lỗi bit (BER)', fontsize=11)
    ax.set_title(f'Đặc tuyến BER - {mod_selected} (R={cr_selected})', fontsize=13)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc='upper right')
    st.pyplot(fig1)

# ==========================================
# NỘI DUNG TAB 2: ĐỀ TÀI 1 (HỆ THỐNG AMC TỰ ĐỘNG)
# ==========================================
with tab2:
    st.header("Hệ Thống Thích Ứng Điều Chế & Mã Hóa Kênh (AMC)")
    st.markdown("Hệ thống tự động sử dụng AI để ra quyết định lựa chọn bộ tham số MCS tối ưu.")
    
    col_t2_1, col_t2_2 = st.columns(2)
    with col_t2_1:
        target_ber = st.select_slider(
            "Ngưỡng BER tối đa cho phép (Target BER):",
            options=[1e-2, 1e-3, 1e-4], value=1e-3,
            format_func=lambda x: f"{x:.1e}"
        )
    with col_t2_2:
        robust_mode = st.checkbox("Kích hoạt chế độ An toàn (Dùng cận trên +1.96σ để chống rớt gói)", value=True)
        
    # Quyết định MCS tối ưu tại Eb/N0 hiện tại
    best_profile = None
    max_achieved_tp = -1.0
    
    for mcs in MCS_PROFILES:
        pt = np.array([[ebno_slider, mcs['bits'], mcs['cr']]])
        m_log, s_log = gpr_model.predict(pt, return_std=True)
        eval_log = (m_log[0] + 1.96 * s_log[0]) if robust_mode else m_log[0]
        p_ber = 10.0 ** eval_log
        
        bler = 1.0 - (1.0 - np.clip(p_ber, 0, 1.0)) ** 1000
        eff_tp = mcs['raw_tp'] * (1.0 - bler)
        
        if p_ber <= target_ber and eff_tp > max_achieved_tp:
            max_achieved_tp = eff_tp
            best_profile = mcs
            
    if best_profile is None:
        best_profile = MCS_PROFILES[0]
        final_tp_val = 0.0
        st.error("⚠️ Sóng quá yếu: Không cấu hình nào thỏa mãn ngưỡng lỗi. Hệ thống hạ về MCS 1 để duy trì kết nối cơ bản.")
    else:
        final_tp_val = max_achieved_tp
        
    m1, m2, m3 = st.columns(3)
    m1.metric("Eb/N0 Kênh truyền", f"{ebno_slider:.1f} dB")
    m2.metric("Quyết định MCS của AI", best_profile['name'])
    m3.metric("Thông lượng khả dụng", f"{final_tp_val:.2f} bps/Hz")
    
    # Biểu đồ Thông lượng & Bậc thang chuyển vùng MCS
    ebno_sweep = np.linspace(-2.0, 20.0, 120)
    sweep_tp = []
    sweep_idx = []
    
    for e in ebno_sweep:
        cur_best_tp = -1.0
        cur_idx = 0
        for idx, mcs in enumerate(MCS_PROFILES):
            pt = np.array([[e, mcs['bits'], mcs['cr']]])
            m_l, s_l = gpr_model.predict(pt, return_std=True)
            eval_l = (m_l[0] + 1.96 * s_l[0]) if robust_mode else m_l[0]
            b_val = 10.0 ** eval_l
            bler_v = 1.0 - (1.0 - np.clip(b_val, 0, 1.0)) ** 1000
            t_val = mcs['raw_tp'] * (1.0 - bler_v)
            if b_val <= target_ber and t_val > cur_best_tp:
                cur_best_tp = t_val
                cur_idx = idx
        sweep_tp.append(max(0.0, cur_best_tp))
        sweep_idx.append(cur_idx)
        
    fig2, (ax_tp, ax_step) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_tp.plot(ebno_sweep, sweep_tp, 'g-', linewidth=2.2, label='Thông lượng AMC tự thích ứng')
    ax_tp.plot(ebno_slider, final_tp_val, 'ro', markersize=9, label='Trạng thái hiện tại')
    ax_tp.set_ylabel('Thông lượng (bps/Hz)', fontsize=11)
    ax_tp.set_title('Đặc tuyến Thông Lượng Tối Ưu và Chuyển Đổi MCS Theo Điều Kiện Sóng', fontsize=12)
    ax_tp.grid(True, linestyle=":", alpha=0.6)
    ax_tp.legend(loc='upper left')
    
    ax_step.step(ebno_sweep, [i + 1 for i in sweep_idx], where='post', color='darkblue', linewidth=2)
    ax_step.axvline(x=ebno_slider, color='red', linestyle='--', alpha=0.7)
    ax_step.set_yticks(range(1, len(MCS_PROFILES) + 1))
    ax_step.set_yticklabels([m['mod'] + f" (R={m['cr']})" for m in MCS_PROFILES])
    ax_step.set_xlabel('$E_b/N_0$ của kênh truyền (dB)', fontsize=11)
    ax_step.set_ylabel('Lựa chọn MCS', fontsize=11)
    ax_step.grid(True, linestyle=":", alpha=0.6)
    
    st.pyplot(fig2)
