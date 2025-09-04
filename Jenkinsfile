pipeline {
    agent any

    environment {
        // Set any environment variables here if needed
        CSV_FILE = "Churn_Modelling.csv"
        MODEL_FILE = "model.joblib"
    }

    stages {
        stage('Install Dependencies') {
            steps {
                echo '📦 Installing dependencies...'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Prepare Data') {
            steps {
                echo '📊 Preparing data...'
                sh 'python main.py --prepare'
            }
        }

        stage('Train Model') {
            steps {
                echo '🤖 Training model...'
                sh 'python main.py --train'
            }
        }

        stage('Evaluate Model') {
            steps {
                echo '📈 Evaluating model...'
                sh 'python main.py --evaluate'
            }
        }

        stage('Save Model') {
            steps {
                echo '💾 Saving model...'
                sh 'python main.py --save'
            }
        }

        stage('Load Model & Evaluate') {
            steps {
                echo '🔄 Loading and evaluating model...'
                sh 'python main.py --load'
            }
        }

        stage('Post-Training Checks') {
            parallel {
                stage('Linting') {
                    steps {
                        echo '🔍 Checking code quality...'
                        sh 'flake8 .'
                    }
                }

                stage('Security') {
                    steps {
                        echo '🔐 Checking security...'
                        sh 'bandit -r .'
                    }
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                echo '🧹 Cleaning up...'
                sh 'rm -rf __pycache__'
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed.'
        }
    }
}

