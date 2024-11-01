

# 🎬 Movie Recommendation System

A sophisticated movie recommendation engine with an interactive Streamlit interface. Get personalized movie recommendations based on your preferences!

## 🌟 Features

- **Smart Recommendations**: Uses cosine similarity for accurate movie suggestions
- **Multi-language Support**: Supports 40+ languages with automatic ISO code mapping
- **Era-based Filtering**: Find movies from specific decades
- **Watch History**: Tracks viewed movies to avoid repetition
- **Interactive UI**: User-friendly interface built with Streamlit
- **Detailed Movie Info**: Access to movie posters, ratings, and descriptions


## 🎮 Usage

1. Select your preferences:
   - Choose a genre
   - Select a language
   - Pick an era using the slider
   - Click "Recommend a movie"

2. View your personalized movie recommendation with:
   - Movie poster
   - Title and overview
   - Rating and release year
   - Genre information

## 📁 Project Structure

```
movie-recommendation-system/
│
├── app.py                 # Main Streamlit application
├── movies_data.csv        # Movie dataset
├── requirements.txt       # Project dependencies
├── seen_movies.txt       # Watch history tracking
└── README.md             # Project documentation
```

## 🔧 Technical Details

- **Recommendation Engine**: Utilizes cosine similarity for content-based filtering
- **Genre Classification**: Implements DBSCAN clustering for accurate genre categorization
- **Data Preprocessing**: Comprehensive pipeline for cleaning and preparing movie data
- **UI Framework**: Built with Streamlit for interactive user experience

## 📊 Dataset

The system uses a comprehensive movie dataset with the following information:
- Movie titles
- Release dates
- Original languages
- Genres
- Vote counts and averages
- Popularity metrics
- Movie overviews
- Poster URLs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📧 Contact

Mohamed Ehab - m.ehab1441@gmail.com
Project Link: (https://github.com/ehab1441/Movies-Recommendation-App)

---
⭐️ If you found this project helpful, please give it a star!
