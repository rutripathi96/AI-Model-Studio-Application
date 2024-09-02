import streamlit as st
import pandas as pd
import numpy as np
import time

from io import StringIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.title("AI Model Studio")
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Model Training", "Visualizations"])

# Page 1: Model Training
if page == "Model Training":
    st.title("Model Training Page")

    # App title


    # Sidebar for file upload
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    # Step 1: Data Upload and Exploration
    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the data preview
        st.write("### Data Preview")
        st.write(data.head())

        # Show basic statistics of the dataset
        st.write("### Basic Statistics")
        st.write(data.describe())

        # Show data types and missing values
        buffer = StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.write("### Data Info")
        st.text(s)  # Use st.text() to display the captured output
    else:
        st.write("### Please upload a CSV file to begin.")

    if uploaded_file is not None:
        # Display problem type selection
        st.sidebar.header("Define the Problem")
        problem_type = st.sidebar.selectbox("Select the Problem Type", ["Classification", "Regression", "Clustering"])

        # Allow users to select target and feature columns
        st.sidebar.header("Select Features and Target")
        columns = data.columns.tolist()

        if problem_type != "Clustering":
            # Target column selection for supervised learning
            target_column = st.sidebar.selectbox("Select Target Column", columns)

            # Feature column selection
            feature_columns = st.sidebar.multiselect("Select Feature Columns", columns, default=columns)

        # Proceed only if columns are selected
        if feature_columns:
            X = data[feature_columns]
            if problem_type != "Clustering":
                y = data[target_column]

            # Display selected columns
            st.write("### Selected Features")
            st.write(X.head())

            if problem_type != "Clustering":
                st.write("### Selected Target")
                st.write(y.head())

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    # Sidebar for algorithm selection
    if uploaded_file is not None and feature_columns:
        st.sidebar.header("Choose Machine Learning Algorithm")

        if problem_type == "Classification":
            model_choice = st.sidebar.selectbox("Select Algorithm",
                                                ["Logistic Regression", "Naive Bayes", "K-Nearest Neighbors",
                                                 "Decision Tree", "Random Forest"])
        elif problem_type == "Regression":
            model_choice = st.sidebar.selectbox("Select Algorithm", ["Linear Regression"])
        elif problem_type == "Clustering":
            model_choice = st.sidebar.selectbox("Select Algorithm", ["K-Means Clustering"])

        # Hyperparameter options based on model choice
        st.sidebar.header("Set Hyperparameters")

        if model_choice == "Logistic Regression":
            solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear"])
            C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, step=0.01, value=1.0)
            model = LogisticRegression(solver=solver, C=C)
            # here C is the lambda parameter value in the ridge and lasso regression used for preventing overfitting
            # solver is basically seleting the optimizer or the convergence algorithm that tweaks the parameters in order to minimize the cost function

        elif model_choice == "Naive Bayes":
            model = GaussianNB()  # No hyperparameters to set in the basic Naive Bayes

        elif model_choice == "K-Nearest Neighbors":
            n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20, step=1, value=5)
            weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            # weights Parameter in KNN: The weights parameter in K-Nearest Neighbors specifies how the algorithm assigns importance to the neighbors when making a prediction.
            # if we choose uniform then that means we dont consider distance and every point has equal weight
            # it works when the dataset is unifrmly distributed and the very unbiased
            # by default in sklearn the distance used is euclidain
            # if we choose distance then weights are assigned on distance basis
            # the closer the point the higher the weight

        elif model_choice == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, step=1, value=5)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, step=1, value=2)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            # Let's assume we set min_samples_split = 3. This means any node (subset of data) must have at least 3 samples to be eligible for further splitting.

        elif model_choice == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 100, step=10, value=50)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, step=1, value=5)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, step=1, value=2)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split)

        elif model_choice == "Linear Regression":
            model = LinearRegression()  # No hyperparameters for basic linear regression

        elif model_choice == "K-Means Clustering":
            n_clusters = st.sidebar.slider("Number of Clusters (K)", 1, 10, step=1, value=3)
            init = st.sidebar.selectbox("Initialization Method", ["k-means++", "random"])
            max_iter = st.sidebar.slider("Maximum Iterations", 100, 500, step=50, value=300)
            model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter)

        # Train-Test Split
        if problem_type != "Clustering":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"### Training Set Size: {X_train.shape[0]} samples")
        st.write(f"### Test Set Size: {X_test.shape[0]} samples")
        # Initialize a session state variable to track if the model is trained
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False

        # Train the model
        if st.sidebar.button("Train Model"):
            start_time = time.time()  # Start time

            if problem_type == "Clustering":
                model.fit(X)
                st.write(f"### Model trained with {n_clusters} clusters.")
                st.session_state.model_trained = True  # Set the flag to True after training
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Display results based on problem type
                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"### Model Accuracy: {accuracy:.2f}")
                elif problem_type == "Regression":
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"### Mean Squared Error: {mse:.2f}")
                    st.write(f"### R-Squared Score: {r2:.2f}")

                end_time = time.time()  # End time
                st.write(f"### Training Time: {end_time - start_time:.2f} seconds")

                # Set the flag to True after training
                st.session_state.model_trained = True

            import matplotlib.pyplot as plt  # Import matplotlib for plotting

            import matplotlib.pyplot as plt  # Import matplotlib for plotting
            import numpy as np  # Import numpy for array manipulation

            # Display evaluation metrics
            from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, \
                f1_score

            # Display evaluation metrics
            if problem_type == "Classification":
                st.write("### Confusion Matrix")
                st.write(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

                # Calculate additional classification metrics
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Display classification metrics
                st.write(f"### Precision: {precision:.2f}")
                st.write(f"### Recall: {recall:.2f}")
                st.write(f"### F1 Score: {f1:.2f}")

            if problem_type == "Regression":
                st.write("### Scatter Plot with Best-Fit Line")

                # Create scatter plot
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5, label='Actual Data Points')

                # Calculate the best-fit line
                m, b = np.polyfit(y_test, y_pred, 1)  # m is slope, b is intercept
                ax.plot(y_test, m * y_test + b, color='red', label='Best-Fit Line')

                # Add labels and title
                ax.set_xlabel("True Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Scatter Plot of True vs Predicted Values with Best-Fit Line")
                ax.legend()

                # Display the plot in Streamlit
                st.pyplot(fig)

                # Calculate regression metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Calculate Adjusted R-squared
                n = len(y_test)  # Number of samples
                p = X.shape[1]  # Number of features
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

                # Display regression metrics
                st.write(f"### Mean Squared Error: {mse:.2f}")
                st.write(f"### R-Squared: {r2:.2f}")
                st.write(f"### Adjusted R-Squared: {adj_r2:.2f}")

            import shap

            # Ensure the SHAP visualizer is compatible with the environment
            if problem_type == "Regression":
                st.write("### SHAP Values for Regression")

                # Create a SHAP explainer using the trained model
                explainer = shap.Explainer(st.session_state.model, X_train)

                # Compute SHAP values for the test set
                shap_values = explainer(X_test)

                # Plot the SHAP values summary
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                st.pyplot(fig)

    # Add your model training functionalities

# Page 2: Visualizations
elif page == "Visualizations":
    st.title("Visualization Page")

    # Sidebar for file upload
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    # Step 1: Data Upload and Exploration
    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the data preview
        st.write("### Data Preview")
        st.write(data.head())

        # Select features for visualization
        st.write("Choose features for visualization:")
        features = data.columns.tolist()
        x_feature = st.selectbox("Select X-axis feature", features)
        y_feature = st.selectbox("Select Y-axis feature", features)

        # Visualization options
        viz_type = st.selectbox("Visualization Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram"])

        if viz_type == "Scatter Plot":
            st.write("Scatter Plot Visualization")
            fig, ax = plt.subplots()
            ax.scatter(data[x_feature], data[y_feature], alpha=0.5)
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f"Scatter Plot of {x_feature} vs {y_feature}")
            st.pyplot(fig)

        elif viz_type == "Line Chart":
            st.write("Line Chart Visualization")
            fig, ax = plt.subplots()
            ax.plot(data[x_feature], data[y_feature])
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f"Line Chart of {x_feature} vs {y_feature}")
            st.pyplot(fig)

        elif viz_type == "Bar Chart":
            st.write("Bar Chart Visualization")
            fig, ax = plt.subplots()
            ax.bar(data[x_feature], data[y_feature])
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f"Bar Chart of {x_feature} vs {y_feature}")
            st.pyplot(fig)

        elif viz_type == "Histogram":
            st.write("Histogram Visualization")
            # Allow users to select only one feature for histogram
            hist_feature = st.selectbox("Select feature for Histogram", features)
            fig, ax = plt.subplots()
            ax.hist(data[hist_feature], bins=10, alpha=0.5)
            ax.set_xlabel(hist_feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f"Histogram of {hist_feature}")
            st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to begin.")
