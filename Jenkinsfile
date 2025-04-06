pipeline{
    agent any

    environment {
		SONAR_PROJECT_KEY = 'LLMOPS'
		SONAR_SCANNER_HOME = tool 'SonarqubeScanner'
	}

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/data-guru0/LLMOPS-1-MEDICAL-CHATBOT.git']])
                }
            }
        }

        stage('SonarQube Analysis'){
			steps {
				withCredentials([string(credentialsId: 'sonarqube-token', variable: 'SONAR_TOKEN')]) {
    					
					withSonarQubeEnv('Sonarqube') {
    						sh """
						${SONAR_SCANNER_HOME}/bin/sonar-scanner \
						-Dsonar.projectKey=${SONAR_PROJECT_KEY} \
						-Dsonar.sources=. \
						-Dsonar.host.url=http://sonarqube-dind:9000 \
						-Dsonar.login=${SONAR_TOKEN}
						"""
					}
				}
			}
		}
        
    }
}