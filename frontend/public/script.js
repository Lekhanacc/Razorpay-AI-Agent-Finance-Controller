const messages = document.getElementById("messages");

const userInput = document.getElementById("userInput");

const sendButton = document.getElementById("sendButton");

const loading = document.getElementById("loading");

// Empty means same origin, which is the default when FastAPI serves this UI.
// For a separately hosted frontend, set apiBaseUrl in config.js instead of
// embedding a backend URL throughout the application.
const apiBaseUrl = (window.RAZORPAY_AI_CONFIG?.apiBaseUrl || "").replace(/\/$/, "");
const conversationId = window.crypto?.randomUUID
    ? window.crypto.randomUUID()
    : `razorpay-chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;


function addMessage(text, type, sources = []) {

    const message = document.createElement("div");

    message.className = `message ${type}`;


    const avatar = document.createElement("div");

    avatar.className = "avatar";

    avatar.textContent =
        type === "user" ? "You" : "AI";


    const bubble = document.createElement("div");

    bubble.className = "bubble";


    const title = document.createElement("div");

    title.className = "message-title";

    title.textContent =
        type === "user"
            ? "You"
            : "Razorpay AI Agent";


    const content = document.createElement("p");

    content.textContent = text;


    bubble.appendChild(title);

    bubble.appendChild(content);

    if (type === "bot" && sources.length) {
        const sourceDetails = document.createElement("details");
        sourceDetails.className = "sources";

        const sourceSummary = document.createElement("summary");
        sourceSummary.textContent = `Sources (${sources.length})`;
        sourceDetails.appendChild(sourceSummary);

        const sourceList = document.createElement("ul");
        for (const source of sources) {
            const item = document.createElement("li");
            const link = document.createElement("a");
            link.href = source.url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = source.title || source.source_id || "Razorpay documentation";
            item.appendChild(link);
            sourceList.appendChild(item);
        }
        sourceDetails.appendChild(sourceList);
        bubble.appendChild(sourceDetails);
    }


    message.appendChild(avatar);

    message.appendChild(bubble);


    messages.appendChild(message);


    messages.scrollTop = messages.scrollHeight;
}


async function sendMessage() {

    const input = userInput.value.trim();


    if (!input) {
        return;
    }


    addMessage(input, "user");


    userInput.value = "";

    sendButton.disabled = true;

    loading.classList.remove("hidden");


    try {

        const response = await fetch(`${apiBaseUrl}/v1/rag/answer`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: input,
                conversation_id: conversationId
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.message || "Something went wrong"
            );

        }


        const fallbackAnswer = data.status === "GENERATION_UNAVAILABLE"
            ? "I found grounded Razorpay documentation, but answer generation is not configured on this server. Please configure the optional generation provider, or review the sources below."
            : data.status === "GENERATION_FAILED"
                ? "I found relevant Razorpay documentation, but generation is temporarily unavailable. Please try again or review the sources below."
                : "I couldn't provide an answer for that request.";

        addMessage(data.answer || fallbackAnswer, "bot", data.sources || []);


    } catch (error) {

        console.error(error);

        addMessage(
            "Sorry, something went wrong. Please try again.",
            "bot"
        );

    } finally {

        loading.classList.add("hidden");

        sendButton.disabled = false;

        userInput.focus();

    }
}


sendButton.addEventListener(
    "click",
    sendMessage
);


userInput.addEventListener(
    "keydown",
    (event) => {

        if (event.key === "Enter") {
            sendMessage();
        }

    }
);
