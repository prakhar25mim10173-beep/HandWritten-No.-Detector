import streamlit as st
from PIL import Image
import cv2

from predict import predict_digit, predict_digits
from equation import solve_equation, format_equation


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Handwritten Math AI",
    page_icon="🧮",
    layout="wide"
)


# HEADER

st.title("🧮 Handwritten Math AI")

st.write(
    "An AI-powered system for handwritten digit recognition "
    "and basic mathematical expression solving."
)

st.divider()


# SIDEBAR

st.sidebar.title("⚙️ Recognition Mode")

mode = st.sidebar.radio(
    "Choose a mode:",
    [
        "🔢 Single Digit",
        "🔢 Multiple Digits",
        "🧮 Equation Solver"
    ]
)


# SINGLE DIGIT MODE

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


# MULTIPLE DIGIT MODE

elif mode == "🔢 Multiple Digits":

    st.header("🔢 Multiple Digit Recognition")

    st.write(
        "Upload one image containing multiple handwritten digits."
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"]
    )

    show_debug = st.checkbox(
        "Show segmentation debug view "
        "(binary image, bounding boxes, per-digit crops)",
        value=False
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

                results, debug_info = predict_digits(image, debug=show_debug)

                # DEBUG PANEL
                if show_debug and debug_info is not None:

                    st.subheader("🔍 Segmentation Debug")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Full binarized image**")
                        st.image(
                            debug_info["binary_full"],
                            use_container_width=True,
                            clamp=True
                        )

                    with col2:
                        st.markdown("**Detected writing line (after closing)**")
                        st.image(
                            debug_info["line_closed"],
                            use_container_width=True,
                            clamp=True
                        )

                    st.markdown("**Bounding boxes on full image**")
                    st.image(
                        cv2.cvtColor(debug_info["boxes_drawn"], cv2.COLOR_BGR2RGB),
                        use_container_width=True
                    )

                    st.write(
                        f"Raw connected components found: "
                        f"**{debug_info['num_raw_components']}**  \n"
                        f"Final digit boxes after filtering + merging: "
                        f"**{debug_info['num_final_digits']}**"
                    )

                    if debug_info["num_raw_components"] != debug_info["num_final_digits"]:
                        st.info(
                            "Raw components and final boxes differ, meaning the "
                            "area/height filtering or box-merging step changed the "
                            "count. If the final number still doesn't match what you "
                            "wrote, that tells us which stage to adjust next."
                        )

                    if debug_info.get("crops"):
                        st.markdown("**Individual crops sent to the CNN**")
                        crop_cols = st.columns(len(debug_info["crops"]))
                        for i, crop in enumerate(debug_info["crops"]):
                            with crop_cols[i]:
                                st.image(
                                    crop,
                                    caption=f"Crop {i + 1}",
                                    use_container_width=True,
                                    clamp=True
                                )

                    st.divider()

                # NORMAL RESULTS

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

# EQUATION MODE

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

# FOOTER

st.divider()

st.caption(
    "Handwritten Math AI | Machine Learning Project"
)
