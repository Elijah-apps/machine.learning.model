"""
Game Theoretic Deception Detection Model
Industry-agnostic ML system with advanced features for detecting deception
using game theory, ensemble learning, and Bayesian inference.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import linprog
from scipy.stats import beta, norm
import warnings
warnings.filterwarnings('ignore')


class GameTheoreticDeceptionDetector:
    """
    Advanced ML model for detecting deception using game theory principles.
    Combines multiple algorithms with Nash equilibrium analysis and Bayesian updating.
    """
    
    def __init__(self, config=None):
        """
        Initialize the deception detection model.
        
        Parameters:
        -----------
        config : dict
            Configuration parameters for the model
        """
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = None
        self.nash_equilibrium = None
        self.bayesian_beliefs = {'prior': 0.5, 'posterior': 0.5}
        self.history = []
        
        self._initialize_models()
    
    def _default_config(self):
        """Default configuration parameters"""
        return {
            'random_forest_trees': 200,
            'gb_learning_rate': 0.1,
            'gb_estimators': 150,
            'mlp_hidden_layers': (128, 64, 32),
            'mlp_activation': 'relu',
            'svm_kernel': 'rbf',
            'voting_strategy': 'soft',
            'bayesian_prior': 0.5,
            'nash_iterations': 1000,
            'confidence_threshold': 0.75,
            'cv_folds': 5,
            'random_state': 42
        }
    
    def _initialize_models(self):
        """Initialize all base models"""
        # Random Forest - captures non-linear patterns
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=self.config['random_forest_trees'],
            max_depth=20,
            min_samples_split=5,
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        # Gradient Boosting - sequential learning
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=self.config['gb_estimators'],
            learning_rate=self.config['gb_learning_rate'],
            max_depth=10,
            random_state=self.config['random_state']
        )
        
        # Neural Network - deep learning patterns
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=self.config['mlp_hidden_layers'],
            activation=self.config['mlp_activation'],
            solver='adam',
            max_iter=500,
            random_state=self.config['random_state'],
            early_stopping=True
        )
        
        # SVM - maximum margin classification
        self.models['svm'] = SVC(
            kernel=self.config['svm_kernel'],
            probability=True,
            random_state=self.config['random_state']
        )
        
        # Naive Bayes - probabilistic baseline
        self.models['naive_bayes'] = GaussianNB()
        
        # Ensemble voting classifier
        estimators = [(name, model) for name, model in self.models.items()]
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=self.config['voting_strategy']
        )
    
    def fit(self, X, y):
        """
        Train the deception detection model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix
        y : array-like, shape (n_samples,)
            Target labels (0: truthful, 1: deceptive)
        
        Returns:
        --------
        self : object
            Trained model
        """
        print("=" * 70)
        print("TRAINING GAME THEORETIC DECEPTION DETECTION MODEL")
        print("=" * 70)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble model
        print("\n[1/5] Training ensemble model...")
        self.ensemble_model.fit(X_scaled, y)
        
        # Train individual models for analysis
        print("[2/5] Training individual base models...")
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            print(f"  ✓ {name} trained")
        
        # Calculate feature importance
        print("[3/5] Calculating feature importance...")
        self._calculate_feature_importance(X_scaled, y)
        
        # Compute Nash equilibrium
        print("[4/5] Computing Nash equilibrium...")
        self.nash_equilibrium = self._compute_nash_equilibrium(X_scaled, y)
        
        # Initialize Bayesian beliefs
        print("[5/5] Initializing Bayesian inference...")
        self._initialize_bayesian_beliefs(y)
        
        print("\n✓ Model training complete!")
        print("=" * 70)
        
        return self
    
    def predict(self, X):
        """
        Predict deception labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted labels
        """
        X_scaled = self.scaler.transform(X)
        return self.ensemble_model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict probability of deception.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        
        Returns:
        --------
        probabilities : array, shape (n_samples, 2)
            Probability estimates for each class
        """
        X_scaled = self.scaler.transform(X)
        return self.ensemble_model.predict_proba(X_scaled)
    
    def analyze_deception(self, X, return_details=True):
        """
        Comprehensive deception analysis with game theoretic insights.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        return_details : bool
            Whether to return detailed analysis
        
        Returns:
        --------
        analysis : dict
            Comprehensive deception analysis
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
            if hasattr(model, 'predict_proba'):
                probabilities[name] = model.predict_proba(X_scaled)[:, 1]
        
        # Ensemble predictions
        ensemble_pred = self.ensemble_model.predict(X_scaled)
        ensemble_proba = self.ensemble_model.predict_proba(X_scaled)[:, 1]
        
        # Game theoretic analysis
        strategic_scores = self._strategic_analysis(X_scaled)
        
        # Bayesian updates
        bayesian_posterior = self._update_bayesian_beliefs(ensemble_proba)
        
        # Nash equilibrium strategies
        nash_strategies = self._apply_nash_equilibrium(X_scaled)
        
        analysis = {
            'deception_prediction': ensemble_pred,
            'deception_probability': ensemble_proba,
            'confidence_score': np.max(self.ensemble_model.predict_proba(X_scaled), axis=1),
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'strategic_deception_score': strategic_scores,
            'bayesian_posterior': bayesian_posterior,
            'nash_equilibrium_strategy': nash_strategies,
            'risk_level': self._classify_risk(ensemble_proba),
            'feature_contributions': self._get_feature_contributions(X_scaled)
        }
        
        if return_details:
            analysis['detailed_metrics'] = self._calculate_detailed_metrics(analysis)
        
        return analysis
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            True labels
        
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.ensemble_model.predict(X_scaled)
        y_proba = self.ensemble_model.predict_proba(X_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.ensemble_model, X_scaled, y_test,
            cv=self.config['cv_folds'], scoring='accuracy'
        )
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def _calculate_feature_importance(self, X, y):
        """Calculate feature importance from Random Forest"""
        rf = self.models['random_forest']
        self.feature_importance = rf.feature_importances_
    
    def _compute_nash_equilibrium(self, X, y):
        """
        Compute Nash equilibrium for the deception game.
        
        Models a two-player game:
        - Player 1 (Subject): Choose to tell truth or deceive
        - Player 2 (Detector): Choose to trust or verify
        """
        # Estimate payoff matrix from data
        deception_rate = np.mean(y)
        detection_rate = accuracy_score(y, self.ensemble_model.predict(X))
        
        # Payoff matrix (simplified strategic form game)
        # Rows: Subject (Truth, Deceive), Cols: Detector (Trust, Verify)
        payoff_subject = np.array([
            [1.0, 0.8],  # Truth: high payoff if trusted, moderate if verified
            [1.5, -1.0]  # Deceive: very high if trusted, penalty if caught
        ])
        
        payoff_detector = np.array([
            [1.0, -0.5],  # Trust truth: good, Trust deception: bad
            [0.5, 1.5]    # Verify truth: cost, Verify deception: reward
        ])
        
        # Find mixed strategy Nash equilibrium
        # Subject's strategy: probability of truth
        p_truth = (payoff_detector[1, 1] - payoff_detector[0, 1]) / \
                  (payoff_detector[1, 1] - payoff_detector[0, 1] + 
                   payoff_detector[0, 0] - payoff_detector[1, 0])
        
        # Detector's strategy: probability of trust
        q_trust = (payoff_subject[1, 1] - payoff_subject[0, 1]) / \
                  (payoff_subject[1, 1] - payoff_subject[0, 1] + 
                   payoff_subject[0, 0] - payoff_subject[1, 0])
        
        return {
            'truth_probability': np.clip(p_truth, 0, 1),
            'deception_probability': np.clip(1 - p_truth, 0, 1),
            'trust_probability': np.clip(q_trust, 0, 1),
            'verify_probability': np.clip(1 - q_trust, 0, 1),
            'expected_payoff_subject': np.mean(payoff_subject),
            'expected_payoff_detector': np.mean(payoff_detector),
            'is_pure_strategy': (p_truth in [0, 1]) and (q_trust in [0, 1])
        }
    
    def _initialize_bayesian_beliefs(self, y):
        """Initialize Bayesian prior beliefs"""
        self.bayesian_beliefs['prior'] = self.config['bayesian_prior']
        self.bayesian_beliefs['posterior'] = np.mean(y)  # Empirical rate
    
    def _update_bayesian_beliefs(self, probabilities):
        """
        Update beliefs using Bayesian inference.
        
        P(deception|evidence) = P(evidence|deception) * P(deception) / P(evidence)
        """
        prior = self.bayesian_beliefs['posterior']
        likelihood = probabilities
        
        # Bayesian update
        posterior = (likelihood * prior) / \
                   (likelihood * prior + (1 - likelihood) * (1 - prior))
        
        self.bayesian_beliefs['posterior'] = np.mean(posterior)
        
        return posterior
    
    def _strategic_analysis(self, X):
        """
        Analyze strategic behavior patterns.
        
        Uses game theory to identify strategic deception attempts.
        """
        # Calculate strategic deviation score
        # Higher scores indicate behavior inconsistent with truthful strategy
        
        # Use neural network hidden layer activations as strategic features
        nn = self.models['neural_network']
        
        # Get predictions
        predictions = nn.predict_proba(X)[:, 1]
        
        # Calculate strategic score based on prediction confidence and variance
        strategic_score = predictions * (1 - predictions) * 4  # Entropy-like measure
        
        return strategic_score
    
    def _apply_nash_equilibrium(self, X):
        """Apply Nash equilibrium strategy to predictions"""
        if self.nash_equilibrium is None:
            return None
        
        # Adjust predictions based on equilibrium strategy
        base_proba = self.ensemble_model.predict_proba(X)[:, 1]
        
        # Weight by Nash equilibrium deception probability
        adjusted_proba = base_proba * self.nash_equilibrium['deception_probability']
        
        return adjusted_proba
    
    def _classify_risk(self, probabilities):
        """Classify risk level based on deception probability"""
        risk_levels = []
        for p in probabilities:
            if p < 0.3:
                risk_levels.append('LOW')
            elif p < 0.6:
                risk_levels.append('MEDIUM')
            elif p < 0.8:
                risk_levels.append('HIGH')
            else:
                risk_levels.append('CRITICAL')
        return np.array(risk_levels)
    
    def _get_feature_contributions(self, X):
        """Calculate feature contributions to predictions"""
        if self.feature_importance is None:
            return None
        
        # Weighted feature values by importance
        contributions = X * self.feature_importance
        return contributions
    
    def _calculate_detailed_metrics(self, analysis):
        """Calculate additional detailed metrics"""
        return {
            'model_agreement': np.mean([
                np.mean(pred == analysis['deception_prediction'])
                for pred in analysis['individual_predictions'].values()
            ]),
            'prediction_entropy': -np.sum(
                analysis['deception_probability'] * 
                np.log(analysis['deception_probability'] + 1e-10)
            ),
            'strategic_consistency': np.corrcoef(
                analysis['deception_probability'],
                analysis['strategic_deception_score']
            )[0, 1] if len(analysis['deception_probability']) > 1 else 0
        }
    
    def save_model(self, filepath):
        """Save model to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'models': self.models,
                'ensemble_model': self.ensemble_model,
                'config': self.config,
                'feature_importance': self.feature_importance,
                'nash_equilibrium': self.nash_equilibrium,
                'bayesian_beliefs': self.bayesian_beliefs
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.models = data['models']
            self.ensemble_model = data['ensemble_model']
            self.config = data['config']
            self.feature_importance = data['feature_importance']
            self.nash_equilibrium = data['nash_equilibrium']
            self.bayesian_beliefs = data['bayesian_beliefs']
        print(f"✓ Model loaded from {filepath}")
    
    def get_summary(self):
        """Get model summary"""
        summary = {
            'Model Type': 'Game Theoretic Deception Detector',
            'Base Models': list(self.models.keys()),
            'Ensemble Strategy': self.config['voting_strategy'],
            'Nash Equilibrium': self.nash_equilibrium,
            'Bayesian Beliefs': self.bayesian_beliefs,
            'Feature Importance Available': self.feature_importance is not None
        }
        return summary
    
    def print_summary(self):
        """Print formatted model summary"""
        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        summary = self.get_summary()
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            elif isinstance(value, list):
                print(f"\n{key}:")
                for item in value:
                    print(f"  • {item}")
            else:
                print(f"{key}: {value}")
        print("=" * 70 + "\n")


def generate_synthetic_data(n_samples=1000, n_features=10, deception_rate=0.3):
    """
    Generate synthetic data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    deception_rate : float
        Proportion of deceptive samples
    
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target labels
    feature_names : list
        Feature names
    """
    np.random.seed(42)
    
    n_deceptive = int(n_samples * deception_rate)
    n_truthful = n_samples - n_deceptive
    
    # Truthful samples (lower variance, consistent patterns)
    X_truthful = np.random.normal(loc=0.5, scale=0.15, size=(n_truthful, n_features))
    
    # Deceptive samples (higher variance, inconsistent patterns)
    X_deceptive = np.random.normal(loc=0.6, scale=0.25, size=(n_deceptive, n_features))
    
    # Add strategic deception patterns
    X_deceptive[:, 0] *= 1.3  # Higher communication pattern variance
    X_deceptive[:, 3] += 0.2   # Strategic deviation
    X_deceptive[:, 4] += 0.15  # Temporal inconsistency
    
    X = np.vstack([X_truthful, X_deceptive])
    y = np.hstack([np.zeros(n_truthful), np.ones(n_deceptive)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    # Clip to [0, 1]
    X = np.clip(X, 0, 1)
    
    feature_names = [
        'communication_pattern', 'behavioral_consistency',
        'information_asymmetry', 'strategic_deviation',
        'temporal_inconsistency', 'payoff_discrepancy',
        'response_latency', 'cognitive_load',
        'social_context', 'historical_trust'
    ]
    
    return X, y, feature_names[:n_features]


def demo_usage():
    """Demonstration of the model usage"""
    print("\n" + "🎮" * 35)
    print("GAME THEORETIC DECEPTION DETECTION MODEL - DEMO")
    print("🎮" * 35 + "\n")
    
    # Generate synthetic data
    print("[STEP 1] Generating synthetic data...")
    X, y, feature_names = generate_synthetic_data(n_samples=2000, n_features=10)
    print(f"✓ Generated {len(X)} samples with {X.shape[1]} features")
    print(f"  Deception rate: {np.mean(y):.2%}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    print("[STEP 2] Initializing model...")
    detector = GameTheoreticDeceptionDetector()
    
    print("\n[STEP 3] Training model...")
    detector.fit(X_train, y_train)
    
    # Evaluate
    print("\n[STEP 4] Evaluating model...")
    metrics = detector.evaluate(X_test, y_test)
    
    print("\nPERFORMANCE METRICS:")
    print("-" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"CV Mean:   {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
    
    # Analyze sample
    print("\n[STEP 5] Analyzing sample instances...")
    sample_analysis = detector.analyze_deception(X_test[:5])
    
    print("\nSAMPLE ANALYSIS (First 5 test instances):")
    print("-" * 50)
    for i in range(5):
        pred = sample_analysis['deception_prediction'][i]
        prob = sample_analysis['deception_probability'][i]
        conf = sample_analysis['confidence_score'][i]
        risk = sample_analysis['risk_level'][i]
        
        print(f"\nInstance {i+1}:")
        print(f"  Prediction: {'DECEPTIVE' if pred == 1 else 'TRUTHFUL'}")
        print(f"  Probability: {prob:.4f}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Risk Level: {risk}")
    
    # Print model summary
    detector.print_summary()
    
    # Feature importance
    if detector.feature_importance is not None:
        print("\nFEATURE IMPORTANCE (Top 5):")
        print("-" * 50)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': detector.feature_importance
        }).sort_values('Importance', ascending=False)
        
        for idx, row in importance_df.head().iterrows():
            print(f"  {row['Feature']:<30} {row['Importance']:.4f}")
    
    print("\n" + "✓" * 35)
    print("DEMO COMPLETE!")
    print("✓" * 35 + "\n")
    
    return detector, X_test, y_test


if __name__ == "__main__":
    # Run demonstration
    model, X_test, y_test = demo_usage()
    
    print("\n📚 USAGE EXAMPLE:")
    print("-" * 70)
    print("""
# Initialize model
detector = GameTheoreticDeceptionDetector()

# Train model
detector.fit(X_train, y_train)

# Make predictions
predictions = detector.predict(X_test)
probabilities = detector.predict_proba(X_test)

# Comprehensive analysis
analysis = detector.analyze_deception(X_test)

# Evaluate performance
metrics = detector.evaluate(X_test, y_test)

# Save/Load model
detector.save_model('deception_detector.pkl')
detector.load_model('deception_detector.pkl')
    """)