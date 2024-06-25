\documentclass{article}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{Amazon Reviews Sentiment Analysis Web App}
\author{Your Name}
\date{\today}

\begin{document}

% Include the image at the beginning
\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\textwidth]{logo.jpg} % Replace 'logo.png' with your image file name
    \caption*{Amazon Reviews Sentiment Analysis Web App}
    \label{fig:logo}
\end{figure}

\maketitle

\section*{Introduction}
This project is a web application for performing sentiment analysis on Amazon reviews. It uses the NLTK library's VADER sentiment analysis tool to classify reviews as positive, negative, or neutral, and provides a visual representation of the sentiment distribution.

\section*{Features}
\begin{itemize}
    \item Input a review and receive a detailed sentiment analysis.
    \item Visualization of sentiment distribution using a pie chart.
    \item Responsive design using Bootstrap for an attractive and user-friendly interface.
\end{itemize}

\section*{Installation}
To set up the project on your local machine, follow these steps:

\begin{enumerate}
    \item Clone the repository:
    \begin{verbatim}
    git clone https://github.com/yourusername/amazon-reviews-sentiment-analysis.git
    \end{verbatim}

    \item Navigate to the project directory:
    \begin{verbatim}
    cd amazon-reviews-sentiment-analysis
    \end{verbatim}

    \item Install the required libraries:
    \begin{verbatim}
    pip install Flask pandas numpy matplotlib seaborn nltk tqdm
    \end{verbatim}
\end{enumerate}

\section*{Usage}
To run the web application, use the following command:
\begin{verbatim}
python app.py
\end{verbatim}

Then, open your web browser and navigate to \url{http://127.0.0.1:5000/} to use the sentiment analysis tool.

\section*{Project Structure}
\begin{verbatim}
amazon-reviews-sentiment-analysis/
│
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
└── static/
\end{verbatim}

\section*{Templates}
The HTML templates use Bootstrap for styling and responsiveness. The main templates are:
\begin{itemize}
    \item \textbf{index.html}: The home page where users can input their reviews.
    \item \textbf{result.html}: The results page displaying the sentiment analysis and pie chart.
\end{itemize}

\section*{License}
This project is licensed under the MIT License. See the \texttt{LICENSE} file for more details.

\section*{Acknowledgements}
This project uses the following open-source libraries:
\begin{itemize}
    \item Flask
    \item pandas
    \item numpy
    \item matplotlib
    \item seaborn
    \item nltk
\end{itemize}

\end{document}
