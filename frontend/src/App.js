// src/SmartProjectForm.js
import React, { useState } from 'react';
import {
  Slider, TextField, Typography, MenuItem, Button, Box, Paper, CircularProgress
} from '@mui/material';

const SmartProjectForm = () => {
  const [projectDomain, setProjectDomain] = useState('Finance');
  const [techStack, setTechStack] = useState('react,node,mongo');
  const [deliveryTime, setDeliveryTime] = useState(15);
  const [complexity, setComplexity] = useState('Low');
  const [clientRating, setClientRating] = useState(4.5);
  const [projectSize, setProjectSize] = useState(400);
  const [urgency, setUrgency] = useState('Low');
  const [teamPerformance, setTeamPerformance] = useState(4.0);
  const [teamWorkload, setTeamWorkload] = useState(2);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    const payload = {
      project_domain: projectDomain.toLowerCase(),
      tech_stack: techStack.split(',').map(s => s.trim()),
      delivery_time: deliveryTime,
      project_complexity: complexity.toLowerCase(),
      client_rating: clientRating,
      project_size: projectSize,
      deadline_urgency: urgency.toLowerCase(),
      team_performance: teamPerformance,
      team_workload: teamWorkload,
    };

    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Backend error');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{   minHeight: '100vh',
    background: 'linear-gradient(to bottom,rgba(41, 40, 40, 0.12),rgba(97, 81, 81, 0.29),rgba(116, 114, 114, 0.72))',
    padding: 4,
    fontFamily: 'Poppins, sans-serif',
    color: 'white'}}>
    <Box sx={{ width: '85%', margin: 'auto', color: 'white' }}>
      <Typography variant="h4" gutterBottom fontWeight="bold" fontFamily="Arial">Smart Project Assignment</Typography>

      <TextField select label="Project Domain" value={projectDomain} onChange={(e) => setProjectDomain(e.target.value)}
        fullWidth margin="normal" sx={{ '& label': { color: 'white' },
    '& .MuiInputBase-root': {
      backgroundColor: '#7c787849',
      borderRadius: '5px',
    },
    '& .MuiInputBase-input': {
      color: 'white',
    },
    '& .MuiOutlinedInput-notchedOutline': {
      borderColor: 'white', } }}
      >
        {['Finance', 'Healthcare', 'E-commerce', 'Education', 'Logistics', 'Travel','Gaming','Real Estate','Manufacturing','Social Media'].map((opt) => (
          <MenuItem key={opt} value={opt}>{opt}</MenuItem>
        ))}
      </TextField>

      <TextField
        label="Tech Stack (comma-separated)"
        value={techStack}
        onChange={(e) => setTechStack(e.target.value)}
        fullWidth margin="normal"
        sx={{ '& label': { color: 'white' },
    '& .MuiInputBase-root': {
      backgroundColor: '#7c787849',
      borderRadius: '5px',
    },
    '& .MuiInputBase-input': {
      color: 'white',
    },
    '& .MuiOutlinedInput-notchedOutline': {
      borderColor: 'white', } }}
      />

      <Typography gutterBottom>Delivery Time (days): {deliveryTime}</Typography>
      <Slider value={deliveryTime} onChange={(e, val) => setDeliveryTime(val)} min={7} max={365} step={1} />

      <TextField select label="Complexity" value={complexity} onChange={(e) => setComplexity(e.target.value)}
        fullWidth margin="normal" sx={{ '& label': { color: 'white' },
    '& .MuiInputBase-root': {
      backgroundColor: '#7c787849',
      borderRadius: '5px',
    },
    '& .MuiInputBase-input': {
      color: 'white',
    },
    '& .MuiOutlinedInput-notchedOutline': {
      borderColor: 'white', } }}
      >
        {['Low', 'Medium', 'High'].map((opt) => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
      </TextField>

      <Typography gutterBottom>Client Rating: {clientRating}</Typography>
      <Slider value={clientRating} onChange={(e, val) => setClientRating(val)} min={3} max={5} step={0.1} />

      <Typography gutterBottom>Project Size (person-hours): {projectSize}</Typography>
      <Slider value={projectSize} onChange={(e, val) => setProjectSize(val)} min={100} max={1000} step={10} />

      <TextField select label="Urgency" value={urgency} onChange={(e) => setUrgency(e.target.value)}
        fullWidth margin="normal" sx={{ '& label': { color: 'white' },
    '& .MuiInputBase-root': {
      backgroundColor: '#7c787849',
      borderRadius: '5px',
    },
    '& .MuiInputBase-input': {
      color: 'white',
    },
    '& .MuiOutlinedInput-notchedOutline': {
      borderColor: 'white', }}}
      >
        {['Low', 'Medium', 'High'].map((opt) => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
      </TextField>

      <Typography gutterBottom>Team Performance: {teamPerformance}</Typography>
      <Slider value={teamPerformance} onChange={(e, val) => setTeamPerformance(val)} min={3} max={5} step={0.1} />

      <Typography gutterBottom>Current Team Workload: {teamWorkload}</Typography>
      <Slider value={teamWorkload} onChange={(e, val) => setTeamWorkload(val)} min={0} max={10} step={1} />

      <Button variant="contained" color="primary" fullWidth onClick={handleSubmit}>
        Recommend
      </Button>

      {loading && <CircularProgress sx={{ mt: 2 }} />}

      {error && <Typography color="error" mt={2}>{error}</Typography>}

      {result && (
        <Paper elevation={3} sx={{ mt: 4, p: 3, backgroundColor: 'rgba(255, 255, 255, 0.05)', color: 'white' }}>
          <Typography variant="h6">✅ Recommended Team: {result.recommended_team}</Typography>
          <Typography>Confidence: {result.confidence}%</Typography>
          <Typography>Alternate: {result.alternate_team} ({result.alternate_confidence}%)</Typography>

          {result.summary && (
            <>
              <Typography mt={2} fontWeight="bold">🧠 Summary</Typography>
              <Typography>{result.summary}</Typography>
            </>
          )}

          {result.explanation && (
            <>
              <Typography mt={2} fontWeight="bold">📊 Why this team?</Typography>
              <ul>
                {result.explanation.map((item, index) => (
                  <li key={index}>
                    {item.impact > 0 ? '🔺' : '🔻'} {item.feature} — impact: {item.impact}
                  </li>
                ))}
              </ul>
            </>
          )}
        </Paper>
      )}
    </Box>
    </Box>
  );
};

export default SmartProjectForm;
