import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD TRAINED MODEL
# =========================

data = joblib.load("human_activity_model.pkl")

model = data["model"]
label_encoder = data["label_encoder"]


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Human Activity Monitoring",
    page_icon="🏃",
    layout="wide"
)


# =========================
# HEADER
# =========================

st.title("🏃 Human Activity Monitoring")
st.subheader("Early Neurodegenerative Disease Assessment")

st.write(
    "Machine learning prototype for human activity classification "
    "using IMU-like movement data."
)

st.divider()


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🧠 Project Information")

st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Input: IMU Sensor Data / CSV")
st.sidebar.write("Activities: 6")
st.sidebar.write("Model Type: Classification")
st.sidebar.write("Train/Test Split: 80/20")
st.sidebar.write("Sampling Rate: 50 Hz")
st.sidebar.write("Window: 2.56 seconds")

st.sidebar.info(
    "Research prototype only. It is not a medical diagnosis system."
)


# =========================
# CSV UPLOAD
# =========================

st.header("📁 Sensor Data Input")

st.write(
    "Upload an IMU sensor CSV containing acceleration values "
    "for X, Y and Z axes."
)

uploaded_file = st.file_uploader(
    "Upload Sensor CSV",
    type=["csv"]
)


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(ax, ay, az, fs=50):

    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)

    # Acceleration magnitude
    magnitude = np.sqrt(
        ax ** 2 + ay ** 2 + az ** 2
    )

    centered = magnitude - magnitude.mean()

    # Frequency analysis
    fft_values = np.abs(
        np.fft.rfft(centered)
    )

    frequencies = np.fft.rfftfreq(
        len(centered),
        1 / fs
    )

    if len(fft_values) > 1:
        dominant_frequency = frequencies[
            np.argmax(fft_values[1:]) + 1
        ]
    else:
        dominant_frequency = 0

    # Peak detection
    threshold = max(
        centered.std() * 0.5,
        1e-6
    )

    peaks = []

    for i in range(1, len(centered) - 1):
        if (
            centered[i] > centered[i - 1]
            and centered[i] > centered[i + 1]
            and centered[i] > threshold
        ):
            peaks.append(i)

    peak_rate = len(peaks) / (len(magnitude) / fs)

    return [
        magnitude.mean(),
        magnitude.std(),
        magnitude.min(),
        magnitude.max(),
        np.percentile(magnitude, 25),
        np.percentile(magnitude, 75),
        dominant_frequency,
        peak_rate,
        ax.std(),
        ay.std(),
        az.std()
    ]


# =========================
# PROCESS CSV
# =========================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("✅ CSV successfully loaded")

        st.header("📊 Raw Sensor Data")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.write(
            f"Rows detected: **{len(df)}**"
        )

        # Find X/Y/Z columns automatically
        columns_lower = {
            str(col).lower(): col
            for col in df.columns
        }

        x_col = None
        y_col = None
        z_col = None

        for lower_name, original_name in columns_lower.items():

            if x_col is None and (
                lower_name == "x"
                or "accel x" in lower_name
                or "acceleration x" in lower_name
                or "axis x" in lower_name
                or lower_name.endswith("_x")
            ):
                x_col = original_name

            if y_col is None and (
                lower_name == "y"
                or "accel y" in lower_name
                or "acceleration y" in lower_name
                or "axis y" in lower_name
                or lower_name.endswith("_y")
            ):
                y_col = original_name

            if z_col is None and (
                lower_name == "z"
                or "accel z" in lower_name
                or "acceleration z" in lower_name
                or "axis z" in lower_name
                or lower_name.endswith("_z")
            ):
                z_col = original_name

        # Manual selection if automatic detection fails
        st.subheader("🔧 Sensor Columns")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_x = st.selectbox(
                "X-axis column",
                df.columns,
                index=(
                    list(df.columns).index(x_col)
                    if x_col in df.columns
                    else 0
                )
            )

        with col2:
            selected_y = st.selectbox(
                "Y-axis column",
                df.columns,
                index=(
                    list(df.columns).index(y_col)
                    if y_col in df.columns
                    else min(1, len(df.columns) - 1)
                )
            )

        with col3:
            selected_z = st.selectbox(
                "Z-axis column",
                df.columns,
                index=(
                    list(df.columns).index(z_col)
                    if z_col in df.columns
                    else min(2, len(df.columns) - 1)
                )
            )

        # Convert to numeric
        ax = pd.to_numeric(
            df[selected_x],
            errors="coerce"
        )

        ay = pd.to_numeric(
            df[selected_y],
            errors="coerce"
        )

        az = pd.to_numeric(
            df[selected_z],
            errors="coerce"
        )

        valid = pd.DataFrame({
            "X": ax,
            "Y": ay,
            "Z": az
        }).dropna()

        if len(valid) < 20:

            st.error(
                "Not enough numeric sensor data found. "
                "Please upload a CSV containing X, Y and Z acceleration values."
            )

        else:

            ax = valid["X"].values
            ay = valid["Y"].values
            az = valid["Z"].values

            # =========================
            # SENSOR GRAPH
            # =========================

            st.header("📈 Sensor Movement Graph")

            sensor_df = pd.DataFrame({
                "X-axis": ax,
                "Y-axis": ay,
                "Z-axis": az
            })

            st.line_chart(
                sensor_df
            )

            # =========================
            # FEATURE EXTRACTION
            # =========================

            st.header("🧮 Extracted Movement Features")

            features = extract_features(
                ax,
                ay,
                az,
                fs=50
            )

            feature_names = [
                "Mean Acceleration",
                "Movement Variability",
                "Minimum Acceleration",
                "Maximum Acceleration",
                "25th Percentile",
                "75th Percentile",
                "Dominant Frequency",
                "Movement Peak Rate",
                "X-axis Variation",
                "Y-axis Variation",
                "Z-axis Variation"
            ]

            feature_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": features
            })

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

            # =========================
            # PREDICTION
            # =========================

            st.divider()

            if st.button(
                "🔍 Analyze Movement",
                use_container_width=True
            ):

                input_data = np.array(
                    features
                ).reshape(1, -1)

                prediction = model.predict(
                    input_data
                )

                probabilities = model.predict_proba(
                    input_data
                )

                activity = label_encoder.inverse_transform(
                    prediction
                )[0]

                confidence = (
                    np.max(probabilities) * 100
                )

                # =========================
                # RESULT
                # =========================

                st.header("🎯 Activity Prediction")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.success(
                        f"Detected Activity: {activity}"
                    )

                with result_col2:

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                # =========================
                # PROBABILITY
                # =========================

                st.header("📋 Activity Probability")

                probability_df = pd.DataFrame({
                    "Activity": label_encoder.classes_,
                    "Probability (%)": probabilities[0] * 100
                })

                probability_df = probability_df.sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.header("📈 Activity Probability Chart")

                st.bar_chart(
                    probability_df.set_index(
                        "Activity"
                    )["Probability (%)"]
                )

                # =========================
                # MOVEMENT SCREENING
                # =========================

                st.header(
                    "⚠️ Movement Pattern Screening"
                )

                variability = features[1]
                peak_rate = features[7]

                if (
                    variability > 0.8
                    or peak_rate > 7
                ):

                    st.warning(
                        "Higher movement variability "
                        "was detected in this sensor window."
                    )

                    st.write(
                        "This is only a prototype screening "
                        "flag and does not diagnose a disease."
                    )

                else:

                    st.success(
                        "No high-variability movement flag "
                        "detected for this sensor window."
                    )

                # =========================
                # MODEL INFORMATION
                # =========================

                st.header(
                    "🧠 Machine Learning Model"
                )

                model_col1, model_col2, model_col3 = st.columns(3)

                with model_col1:
                    st.metric(
                        "Algorithm",
                        "Random Forest"
                    )

                with model_col2:
                    st.metric(
                        "Activities",
                        "6"
                    )

                with model_col3:
                    st.metric(
                        "Test Accuracy",
                        "99.44%"
                    )

                st.write(
                    "The Random Forest classifier uses "
                    "movement-related features to classify "
                    "human activities."
                )

                # =========================
                # PROJECT PIPELINE
                # =========================

                st.header(
                    "🔄 Project Pipeline"
                )

                st.info(
                    "Sensor Data → Preprocessing → "
                    "Feature Extraction → Random Forest → "
                    "Activity Classification → "
                    "Movement Pattern Screening"
                )

                # =========================
                # SUPPORTED ACTIVITIES
                # =========================

                st.header(
                    "🚶 Supported Activities"
                )

                st.write(
                    ", ".join(
                        label_encoder.classes_
                    )
                )

                # =========================
                # IMPORTANT RESEARCH NOTE
                # =========================

                st.warning(
                    "Current model was trained using "
                    "synthetic IMU-like training data. "
                    "Real labelled wearable-sensor data "
                    "and clinical validation are required "
                    "before making medical conclusions."
                )

    except Exception as e:

        st.error(
            "Unable to process this CSV."
        )

        st.code(
            str(e)
        )

else:

    st.info(
        "👆 Upload an IMU sensor CSV to begin movement analysis."
    )


# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Human Activity Monitoring for Early Neurodegenerative Disease Assessment"
)