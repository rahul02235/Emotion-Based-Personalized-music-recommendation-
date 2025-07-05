from django.urls import path
from . import views
from . import camera

app_name = 'api'

urlpatterns = [
    path('upload_image/', views.upload_image, name='upload_image'),
    path('recommendations/', views.recommendations_view, name='recommendations'),
    path('recommendations/<str:image_id>/', views.get_recommendations, name='get_recommendations'),
    path('favorites/', views.favorites, name='favorites'),
    # Camera endpoints
    path('camera/list/', camera.get_cameras, name='camera_list'),
    path('camera/start/', camera.start_camera, name='camera_start'),
    path('camera/stop/', camera.stop_camera, name='camera_stop'),
    path('camera/feed/', camera.camera_feed, name='camera_feed'),
    path('camera/emotion/', camera.get_current_emotion, name='get_current_emotion'),
] 