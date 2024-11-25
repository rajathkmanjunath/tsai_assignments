let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 1000; // 1 second

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "cleanupEmail") {
    const currentTime = Date.now();
    if (currentTime - lastRequestTime < MIN_REQUEST_INTERVAL) {
      sendResponse({error: "Please wait a moment before trying again."});
      return true;
    }

    lastRequestTime = currentTime;

    const apiKey = '$API_KEY'; // Replace with your actual API key
    const apiUrl = '$API_URL';

    fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [
          {role: "system", content: "You are a helpful assistant that cleans up and improves email drafts."},
          {role: "user", content: `Please clean up and improve the following email draft: ${request.text}`}
        ]
      })
    })
    .then(response => {
      if (!response.ok) {
        return response.text().then(text => {
          throw new Error(`HTTP error! status: ${response.status}, body: ${text}`);
        });
      }
      return response.json();
    })
    .then(data => {
      if (data.error) {
        throw new Error(data.error.message || 'Unknown error occurred');
      }
      sendResponse({cleanedText: data.choices[0].message.content});
    })
    .catch(error => {
      console.error('Error:', error);
      sendResponse({error: `An error occurred while cleaning up the email: ${error.message}`});
    });

    return true; // Indicates that the response will be sent asynchronously
  }
});
