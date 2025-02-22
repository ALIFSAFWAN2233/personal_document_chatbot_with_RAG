document.getElementById("submit-btn").addEventListener("click", async () => {
    const userQuery = document.getElementById("user-input").value;

    console.log("the query: ", userQuery)
    const response = await fetch("http://127.0.0.1:8000/query/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ 
            user_query: userQuery 
        })
    });

    const data = await response.json();
    console.log("Response from FastAPI:", data.response);

});