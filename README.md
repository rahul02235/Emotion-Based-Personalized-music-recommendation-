# Music Recommendation System

This is a web application that recommends music based on uploaded images. The system uses image analysis and music recommendation models to suggest appropriate songs based on the content of the uploaded images.

## Features

- Image upload through drag-and-drop or file selection
- Image analysis using machine learning models
- Music recommendations based on image content
- Audio preview player for recommended songs

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Apply database migrations:
```bash
python manage.py migrate
```

3. Run the development server:
```bash
python manage.py runserver
```

4. Access the application at `http://localhost:8000`

## Project Structure

- `api/` - Django app containing the backend logic
- `static/` - Static files (CSS, JavaScript)
- `templates/` - HTML templates
- `media/` - Uploaded files
- `music_recommender/` - Main project settings

## Integration Points

To integrate your models:

1. In `api/views.py`, replace the TODO comments in the `upload_image` view with your actual model processing code:
   - Add your image analysis model
   - Add your music recommendation model
   - Configure the song preview URL generation

## Technologies Used

- Backend: Django
- Frontend: HTML, CSS, JavaScript
- Database: SQLite (default)
- File Storage: Local filesystem

  ## 📸 Screenshots
  
   Live Emotion Detection via OpenCV and Pretrained CNN
  
  ![image](https://github.com/user-attachments/assets/542cde46-f655-44b9-87b2-3ad1c6fd9fc6)

  User Sign up and Login
  
  ![image](https://github.com/user-attachments/assets/aee01c95-968b-4ed9-94c0-98cc1e26a63b)

  Front-End of the Music Recommendation System
  
  ![image](https://github.com/user-attachments/assets/9ec18efa-1268-48ec-987b-f4ad6f218d02)

 Final Result
 
  ![image](https://github.com/user-attachments/assets/895c3ee7-be4e-44ce-ad3c-289c8353fcd7)



  
