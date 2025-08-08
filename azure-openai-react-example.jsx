import React, { useState } from 'react';

// Azure OpenAI Configuration
const AZURE_CONFIG = {
  endpoint: "https://vinod-m7y6fqof-eastus2.cognitiveservices.azure.com/",
  apiKey: "CxjrfpmQJB9TxEWZSTRzKTDIbqozO3kvx8S6yO0MGnfa8cdQ7HDMJQQJ99BCACHYHv6XJ3w3AAAAACOGevLG",
  deploymentName: "o4-mini",
  apiVersion: "2024-12-01-preview"
};

const AzureOpenAIComponent = () => {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const callAzureOpenAI = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setLoading(true);
    setError('');
    setResponse('');

    try {
      const response = await fetch(`${AZURE_CONFIG.endpoint}openai/deployments/${AZURE_CONFIG.deploymentName}/chat/completions?api-version=${AZURE_CONFIG.apiVersion}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-key': AZURE_CONFIG.apiKey,
        },
        body: JSON.stringify({
          messages: [
            {
              role: 'user',
              content: prompt
            }
          ],
          max_tokens: 1000,
          temperature: 0.7
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResponse(data.choices[0].message.content);
    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error('Azure OpenAI API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h2>Azure OpenAI React Example</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <h3>Configuration:</h3>
        <div style={{ fontSize: '12px', backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '5px' }}>
          <strong>Endpoint:</strong> {AZURE_CONFIG.endpoint}<br />
          <strong>Deployment:</strong> {AZURE_CONFIG.deploymentName}<br />
          <strong>API Version:</strong> {AZURE_CONFIG.apiVersion}
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="prompt" style={{ display: 'block', marginBottom: '5px' }}>
          <strong>Enter your prompt:</strong>
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your message here..."
          style={{
            width: '100%',
            height: '100px',
            padding: '10px',
            borderRadius: '5px',
            border: '1px solid #ccc',
            fontSize: '14px'
          }}
        />
      </div>

      <button
        onClick={callAzureOpenAI}
        disabled={loading}
        style={{
          backgroundColor: '#0078d4',
          color: 'white',
          padding: '10px 20px',
          border: 'none',
          borderRadius: '5px',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontSize: '16px',
          marginBottom: '20px'
        }}
      >
        {loading ? 'Sending...' : 'Send to Azure OpenAI'}
      </button>

      {error && (
        <div style={{
          backgroundColor: '#ffebee',
          color: '#c62828',
          padding: '10px',
          borderRadius: '5px',
          marginBottom: '20px'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
        <div style={{ marginTop: '20px' }}>
          <h3>Response:</h3>
          <div style={{
            backgroundColor: '#f8f9fa',
            padding: '15px',
            borderRadius: '5px',
            border: '1px solid #dee2e6',
            whiteSpace: 'pre-wrap',
            fontSize: '14px'
          }}>
            {response}
          </div>
        </div>
      )}
    </div>
  );
};

export default AzureOpenAIComponent; 