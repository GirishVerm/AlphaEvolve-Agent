import React, { useState } from 'react';

// Your Azure OpenAI Configuration
const AZURE_CONFIG = {
  endpoint: "https://vinod-m7y6fqof-eastus2.cognitiveservices.azure.com/",
  apiKey: "CxjrfpmQJB9TxEWZSTRzKTDIbqozO3kvx8S6yO0MGnfa8cdQ7HDMJQQJ99BCACHYHv6XJ3w3AAAAACOGevLG",
  deploymentName: "o4-mini",
  apiVersion: "2024-12-01-preview"
};

const SimpleAzureComponent = () => {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const callAzureOpenAI = async () => {
    setLoading(true);
    
    try {
      const result = await fetch(
        `${AZURE_CONFIG.endpoint}openai/deployments/${AZURE_CONFIG.deploymentName}/chat/completions?api-version=${AZURE_CONFIG.apiVersion}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'api-key': AZURE_CONFIG.apiKey,
          },
          body: JSON.stringify({
            messages: [{ role: 'user', content: message }],
            max_tokens: 1000,
            temperature: 0.7
          })
        }
      );

      const data = await result.json();
      setResponse(data.choices[0].message.content);
    } catch (error) {
      console.error('Error:', error);
      setResponse('Error calling Azure OpenAI');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Azure OpenAI React Example</h2>
      
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Enter your message..."
        style={{ width: '300px', padding: '8px', marginRight: '10px' }}
      />
      
      <button 
        onClick={callAzureOpenAI}
        disabled={loading}
        style={{ padding: '8px 16px' }}
      >
        {loading ? 'Sending...' : 'Send'}
      </button>
      
      {response && (
        <div style={{ marginTop: '20px' }}>
          <h3>Response:</h3>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
};

export default SimpleAzureComponent; 