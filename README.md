<h1>🛡️ Fake News Detection System</h1>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<p>An intelligent web application that uses Artificial Neural Networks (ANN) and Natural Language Processing (NLP) to detect fake news articles in real-time. This system achieves <strong>86-91% accuracy</strong> while maintaining <strong>sub-100ms inference latency</strong>, making it suitable for practical deployment.</p>

<div class="badges">
    <span class="badge badge-blue">Python 3.8+</span>
    <span class="badge badge-orange">TensorFlow 2.x</span>
    <span class="badge badge-green">Flask 2.x</span>
    <span class="badge badge-blue">MIT License</span>
</div>

<h2><span class="emoji">🎯</span>Features</h2>

<ul class="feature-list">
    <li><strong>Real-time Detection</strong>: Instant classification of news articles as REAL or FAKE</li>
    <li><strong>High Accuracy</strong>: 86-91% accuracy using optimized ANN architecture</li>
    <li><strong>Confidence Scoring</strong>: Provides percentage-based confidence metrics</li>
    <li><strong>Suspicious Word Detection</strong>: Identifies clickbait and sensationalist keywords</li>
    <li><strong>User-Friendly Interface</strong>: Intuitive web UI with Bootstrap 5 and animated visualizations</li>
    <li><strong>Fast Inference</strong>: Sub-100ms response time on standard CPU infrastructure</li>
    <li><strong>Robust Preprocessing</strong>: Comprehensive text cleaning and validation pipeline</li>
</ul>

<h2><span class="emoji">🏗️</span>Architecture</h2>

<p>The system employs a three-tier architecture:</p>

<ol>
    <li><strong>Presentation Layer</strong>: HTML5, Bootstrap 5, JavaScript</li>
    <li><strong>Application Layer</strong>: Flask web framework</li>
    <li><strong>Model Layer</strong>: TensorFlow/Keras ANN with TF-IDF vectorization</li>
</ol>

<h3>Model Architecture</h3>

<div class="info-box">
    <ul>
        <li><strong>Input</strong>: 10,000-dimensional TF-IDF feature vector (unigrams + bigrams)</li>
        <li><strong>Hidden Layer 1</strong>: 64 neurons with ReLU activation + 50% Dropout</li>
        <li><strong>Hidden Layer 2</strong>: 32 neurons with ReLU activation + 30% Dropout</li>
        <li><strong>Output Layer</strong>: Single neuron with Sigmoid activation</li>
        <li><strong>Loss Function</strong>: Binary Cross-Entropy</li>
        <li><strong>Optimizer</strong>: Adam</li>
    </ul>
</div>

<div class="architecture-diagram">
Input (10,000 features)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (50%)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Dropout (30%)
    ↓
Output Layer (1 neuron, Sigmoid)
    ↓
Prediction (REAL/FAKE)
</div>

<h2><span class="emoji">🚀</span>Installation</h2>

<h3>Prerequisites</h3>
<ul>
    <li>Python 3.8 or higher</li>
    <li>pip package manager</li>
</ul>

<h3>Setup</h3>

<p><strong>1. Clone the repository</strong></p>
<pre><code>git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection</code></pre>

<p><strong>2. Create a virtual environment</strong></p>
<pre><code>python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate</code></pre>

<p><strong>3. Install dependencies</strong></p>
<pre><code>pip install -r requirements.txt</code></pre>

<p><strong>4. Download the dataset</strong></p>
<ul>
    <li>Place the <code>fake.csv</code> dataset in the project root directory</li>
    <li>Dataset should contain columns: <code>title</code>, <code>text</code>, and <code>label</code></li>
</ul>

<h2><span class="emoji">📊</span>Usage</h2>

<h3>Training the Model</h3>

<p>Run the training script to create the model and vectorizer:</p>

<pre><code>python ann_file.py</code></pre>

<p>This will generate:</p>
<ul>
    <li><code>fake_news_detector_ann_fast.keras</code> - Trained ANN model</li>
    <li><code>tokenizer_fast.pkl</code> - TF-IDF vectorizer</li>
</ul>

<h3>Running the Web Application</h3>

<p>Start the Flask server:</p>

<pre><code>python app.py</code></pre>

<p>Navigate to <code>http://localhost:5000</code> in your browser.</p>

<h3>Command-Line Prediction</h3>

<p>The training script includes an interactive CLI mode:</p>

<pre><code>python ann_file.py</code></pre>

<p>Enter news text when prompted, or type <code>exit</code> to quit.</p>

<h2><span class="emoji">🔬</span>Technical Details</h2>

<h3>Text Preprocessing Pipeline</h3>
<ol>
    <li>Lowercase conversion</li>
    <li>URL and email removal using regex</li>
    <li>Special character filtering</li>
    <li>Whitespace normalization</li>
    <li>Minimum length validation (5 words)</li>
</ol>

<h3>Feature Extraction</h3>
<div class="info-box">
    <ul>
        <li><strong>Method</strong>: TF-IDF Vectorization</li>
        <li><strong>Features</strong>: Top 10,000 most significant terms</li>
        <li><strong>N-gram Range</strong>: (1, 2) - unigrams and bigrams</li>
        <li><strong>Formula</strong>: 
            <pre><code>TF-IDF(t,d) = TF(t,d) × log(N / DF(t))</code></pre>
        </li>
    </ul>
</div>

<h3>Performance Metrics</h3>

<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Test Accuracy</td>
            <td>86-91%</td>
        </tr>
        <tr>
            <td>ROC-AUC Score</td>
            <td>0.90-0.94</td>
        </tr>
        <tr>
            <td>Weighted F1-Score</td>
            <td>0.87-0.92</td>
        </tr>
        <tr>
            <td>Inference Latency</td>
            <td>&lt;100ms per request</td>
        </tr>
    </tbody>
</table>

<h2><span class="emoji">📁</span>Project Structure</h2>

<pre><code>fake-news-detection/
│
├── app.py                          # Flask web application
├── ann_file.py                     # Model training script
├── requirements.txt                # Python dependencies
├── fake.csv                        # Training dataset
│
├── templates/
│   └── index.html                  # Web interface
│
├── static/
│   └── ee896eae...jpg             # Background image
│
├── fake_news_detector_ann_fast.keras  # Trained model
└── tokenizer_fast.pkl              # TF-IDF vectorizer</code></pre>

<h2><span class="emoji">🎨</span>Web Interface</h2>

<p>The web application features:</p>
<ul>
    <li>Responsive design with Bootstrap 5</li>
    <li>Animated confidence bar visualization</li>
    <li>Color-coded predictions (Green for REAL, Red for FAKE)</li>
    <li>Typing animation for prediction text</li>
    <li>Highlighted suspicious keywords</li>
    <li>Clean, modern UI with glassmorphism effects</li>
</ul>

<h2><span class="emoji">🔧</span>Dependencies</h2>

<pre><code>tensorflow>=2.10.0
scikit-learn>=1.1.0
pandas>=1.5.0
numpy>=1.23.0
flask>=2.2.0</code></pre>

<h2><span class="emoji">📈</span>Model Training Details</h2>

<div class="info-box">
    <ul>
        <li><strong>Epochs</strong>: 10</li>
        <li><strong>Batch Size</strong>: 32</li>
        <li><strong>Validation Split</strong>: 10%</li>
        <li><strong>Class Weighting</strong>: Balanced to handle class imbalance</li>
        <li><strong>Regularization</strong>: Dropout layers (0.5 and 0.3)</li>
        <li><strong>Train-Test Split</strong>: 80-20 with stratification</li>
    </ul>
</div>

<h2><span class="emoji">🎯</span>Suspicious Keywords Detected</h2>

<p>The system identifies common clickbait and sensationalist terms:</p>
<ul>
    <li><code>shocking</code>, <code>breaking</code>, <code>exclusive</code></li>
    <li><code>viral</code>, <code>unbelievable</code>, <code>clickbait</code></li>
    <li><code>alert</code>, <code>fake</code>, <code>exposed</code>, <code>secret</code></li>
</ul>

<h2><span class="emoji">🤝</span>Contributing</h2>

<p>Contributions are welcome! Please feel free to submit a Pull Request. For major changes:</p>

<ol>
    <li>Fork the repository</li>
    <li>Create your feature branch (<code>git checkout -b feature/AmazingFeature</code>)</li>
    <li>Commit your changes (<code>git commit -m 'Add some AmazingFeature'</code>)</li>
    <li>Push to the branch (<code>git push origin feature/AmazingFeature</code>)</li>
    <li>Open a Pull Request</li>
</ol>

<h2><span class="emoji">📝</span>Future Enhancements</h2>

<ul class="todo-list">
    <li>Multi-modal verification with metadata integration</li>
    <li>Bi-LSTM/GRU layers for better context understanding</li>
    <li>SHAP/LIME integration for model explainability</li>
    <li>Docker containerization for deployment</li>
    <li>Support for multiple languages</li>
    <li>API rate limiting and authentication</li>
    <li>Integration with external fact-checking databases</li>
</ul>

<h2><span class="emoji">👥</span>Team</h2>

<div class="team-member">
    <strong>ESWARAMUTHU M</strong> - RA2311026040007
</div>
<div class="team-member">
    <strong>KRISHITH T P</strong> - RA2311026040016
</div>
<div class="team-member">
    <strong>AATHITYA A</strong> - RA2311026040021
</div>
<div class="team-member">
    <strong>SENDHAN T U</strong> - RA2311026040023
</div>
<div class="team-member">
    <strong>MOHANA KRISHNAN B</strong> - RA2311026040083
</div>

<div class="info-box" style="margin-top: 20px;">
    <p><strong>Under the guidance of</strong>: Ms. A. Vishnupriya, Assistant Professor</p>
    <p><strong>Institution</strong>: SRM Institute of Science and Technology, Vadapalani Campus</p>
    <p><strong>Department</strong>: Computer Science and Engineering (Emerging Technologies)</p>
    <p><strong>Course</strong>: 21CSE326T - Artificial Neural Networks</p>
</div>

<h2><span class="emoji">📚</span>References</h2>

<ol>
    <li>Ahmed, H., Traore, I., & Saad, S. (2017). <em>Detecting Fake News Using Machine Learning: A Survey</em>. arXiv preprint arXiv:1710.04606.</li>
    <li>Kaliyar, R. K., Goswami, A., & Narang, P. (2021). <em>DeepFake-Net: A Multimodal Deep Learning Approach for Fake News Detection</em>. Neural Computing and Applications, 33(20), 13693-13707.</li>
    <li>Zhou, X., Zhang, D., & Zhao, C. (2020). <em>A Survey on Deep Learning for Fake News Detection</em>. Proceedings of the IEEE Transactions on Knowledge and Data Engineering.</li>
    <li>Wang, Y., et al. (2018). <em>EANN: Event Adversarial Neural Network for Robust Fake News Detection</em>. Proceedings of IJCAI, 4349-4355.</li>
</ol>

<h2><span class="emoji">📄</span>License</h2>

<p>This project is licensed under the MIT License - see the LICENSE file for details.</p>

<h2><span class="emoji">🙏</span>Acknowledgments</h2>

<ul>
    <li>SRM Institute of Science and Technology for providing the platform</li>
    <li>Our guide Ms. A. Vishnupriya for her mentorship</li>
    <li>Department of CSE (Emerging Technologies) for support</li>
    <li>Kaggle community for dataset resources</li>
</ul>

<hr style="border: 1px solid #21262d; margin: 40px 0;">

<div class="warning">
    <p><strong>⚠️ Disclaimer</strong>: This system detects stylistic patterns associated with fake news but does not verify factual claims against external databases. Always cross-reference important information with trusted sources.</p>
</div>

<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #161b22; border-radius: 6px;">
    <p style="font-size: 0.9em; color: #8b949e;">
        Made with ❤️ by the Fake News Detection Team<br>
        © 2025 SRM Institute of Science and Technology
    </p>
</div>

</body>
</html>
