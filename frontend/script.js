const API_URL = "http://127.0.0.1:8000";

const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");


// ================================
// REGISTER
// ================================

if (registerForm) {

    registerForm.addEventListener("submit", async function (event) {

        event.preventDefault();

        const name = document
            .getElementById("registerName")
            .value
            .trim();

        const email = document
            .getElementById("registerEmail")
            .value
            .trim()
            .toLowerCase();

        const password = document
            .getElementById("registerPassword")
            .value;

        const confirmPassword = document
            .getElementById("confirmPassword")
            .value;


        if (password !== confirmPassword) {
            alert("Passwords do not match.");
            return;
        }


        if (password.length < 6) {
            alert("Password must be at least 6 characters.");
            return;
        }


        try {

            const response = await fetch(
                `${API_URL}/register`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        name: name,
                        email: email,
                        password: password
                    })
                }
            );


            const data = await response.json();


            if (!response.ok) {
                alert(data.detail || "Registration failed.");
                return;
            }


            alert(data.message);

            registerForm.reset();

            window.location.href = "login.html";


        } catch (error) {

            console.error("Registration Error:", error);

            alert(
                "Cannot connect to StudyX server.\n\n" +
                "Please make sure the FastAPI backend is running."
            );
        }
    });
}


// ================================
// LOGIN
// ================================

if (loginForm) {

    loginForm.addEventListener("submit", async function (event) {

        event.preventDefault();

        const email = document
            .getElementById("loginEmail")
            .value
            .trim()
            .toLowerCase();

        const password = document
            .getElementById("loginPassword")
            .value;


        if (!email || !password) {
            alert("Please enter email and password.");
            return;
        }


        try {

            const response = await fetch(
                `${API_URL}/login`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        email: email,
                        password: password
                    })
                }
            );


            const data = await response.json();


            if (!response.ok) {
                alert(data.detail || "Login failed.");
                return;
            }


            // Save logged-in user
            localStorage.setItem(
                "studyxUser",
                JSON.stringify(data)
            );


            // Directly go to StudyX home
            window.location.href = "index.html";


        } catch (error) {

            console.error("Login Error:", error);

            alert(
                "Cannot connect to StudyX server.\n\n" +
                "Please make sure the FastAPI backend is running."
            );
        }
    });
}