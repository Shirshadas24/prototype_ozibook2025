import streamlit as st
import requests

st.set_page_config(page_title="AI Team Recommender", page_icon="🤖")


st.title("Smart Project Assignment System")

st.write("Fill in project details below to get a recommended team:")

with st.form("project_form"):
    st.subheader("📋 Project Information")   
    project_domain = st.selectbox("Project Domain", ["finance", "healthcare", "e-commerce","manufacturing", "education", "logistics","travel","social media","gaming","real estate"])
    tech_stack_input = st.text_input("Tech Stack (comma-separated)", "react,node,mongo")
    delivery_time = st.slider("Delivery Time (days)", 7, 365, 15)
    project_complexity = st.selectbox("Complexity", ["low", "medium", "high"])
    client_rating = st.slider("Client Rating", 3.0, 5.0, 4.5, step=0.1)
    project_size = st.slider("Estimated Project Size (person-hours)", 100, 1000, 400)
    deadline_urgency = st.selectbox("Urgency", ["low", "medium", "high"])
    team_performance = st.slider("Team Performance", 3.0, 5.0, 4.0, step=0.1)
    team_workload = st.slider("Additional load for this project ", 0, 10, 2)

    submitted = st.form_submit_button("Recommend ")

if submitted:
    st.info("Based on the provided details, here is recommended team for your project.\n An alternate team is also suggested in case the first team is not available.")

    tech_stack = [tech.strip() for tech in tech_stack_input.split(",") if tech.strip()]
    
    payload = {
        "project_domain": project_domain,
        "tech_stack": tech_stack,
        "delivery_time": delivery_time,
        "project_complexity": project_complexity,
        "client_rating": client_rating,
        "project_size": project_size,
        "deadline_urgency": deadline_urgency,
        "team_performance": team_performance,
        "team_workload": team_workload
    }

    try:
        # Send request to backend
        response = requests.post("http://localhost:5000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(
            f" **Suggested Team:** {result['recommended_team']} \n\n"
            f" **Confidence:** {result['confidence']}%\n\n"
            f" **Alternate:** {result['alternate_team']} ({result['alternate_confidence']}%)"
            )
            

            if 'explanation' in result:
                st.subheader("📊 Why this team?")
                st.write("The following features contributed to the recommendation:")
                st.write("The impact is shown as 🔺 for positive and 🔻 for negative contributions")
                for item in result['explanation']:
                    feature = item['feature']
                    impact = item['impact']
                    arrow = "🔺" if impact > 0 else "🔻"
                    st.write(f"{arrow} **{feature}** contributed with impact: `{impact}`")
            if 'summary' in result:
                st.subheader("🧠 Summary of Reasoning")
                st.write(result['summary'])

        else:
            st.error("❌ Backend responded with error.")
            st.code(response.text)

    except Exception as e:
        st.error("🚫 Could not connect to backend.")
        st.code(str(e))
