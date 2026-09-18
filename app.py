import streamlit as st
from PIL import Image

from predict import predict_digit, predict_digits
from equation import solve_equation, format_equation


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Math AI",
    page_icon="🧮",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧮 Handwritten Math AI")

st.write(
    "An AI-powered system for handwritten digit recognition "
    "and basic mathematical expression solving."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Recognition Mode")

mode = st.sidebar.radio(
    "Choose a mode:",
    [
        "🔢 Single Digit",
        "🔢 Multiple Digits",
        "🧮 Equation Solver"
    ]
)


# ==================================================
# SINGLE DIGIT MODE
# ==================================================

if mode == "🔢 Single Digit":

    st.header("🔢 Single Digit Recognition")

    st.write(
        "Upload an image containing one handwritten digit "
        "from 0 to 9."
    )

    uploaded_file = st.file_uploader(
        "Upload digit image",
        type=["png", "jpg", "jpeg"],
        key="single_digit"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Digit",
            width=250
        )

        if st.button("🔍 Recognize Digit"):

            digit, confidence = predict_digit(image)

            st.success(
                f"### Predicted Digit: {digit}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence}%"
            )


# ==================================================
# MULTIPLE DIGIT MODE
# ==================================================

elif mode == "🔢 Multiple Digits":

    st.header("🔢 Multiple Digit Recognition")

    st.write(
        "Upload one image containing multiple handwritten digits."
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=500
        )

        if st.button("Recognize Digits"):

            try:

                # Get predictions
                results = predict_digits(image)

                st.subheader("Detected Digits")

                number = ""

                for i, result in enumerate(results):

                    digit = result["digit"]
                    confidence = result["confidence"]

                    number += str(digit)

                    st.write(
                        f"Digit {i + 1}: **{digit}** "
                        f"— Confidence: **{confidence}%**"
                    )

                st.success(
                    f"Recognized Number: **{number}**"
                )

            except Exception as e:

                st.error(str(e))


# ==================================================
# EQUATION MODE
# ==================================================

else:

    st.header("🧮 Equation Solver")

    st.write(
        "Enter a mathematical expression using numbers "
        "and basic operators."
    )

    st.info(
        "Supported operators: +, -, ×, ÷"
    )

    equation = st.text_input(
        "Enter equation",
        placeholder="Example: 12 + 7 × 2"
    )

    if st.button("🧮 Solve Equation"):

        if equation.strip() == "":
            st.warning("Please enter an equation.")

        else:

            try:

                result = solve_equation(equation)

                formatted = format_equation(equation)

                st.success(
                    f"### Expression: {formatted}"
                )

                st.success(
                    f"### Result: {result}"
                )

            except ValueError as error:

                st.error(
                    f"❌ {error}"
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Handwritten Math AI | Machine Learning Project"
)
