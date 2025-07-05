from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
from .models import UploadedImage, MusicRecommendation
from .ml_models import process_image_and_get_recommendations
from django.core.cache import cache
import json
import os
import logging
import traceback
import hashlib

logger = logging.getLogger(__name__)

# Create your views here.

@method_decorator(login_required, name='dispatch')
@method_decorator(ensure_csrf_cookie, name='dispatch')
class HomeView(TemplateView):
    template_name = 'index.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['songs'] = MusicRecommendation.objects.filter(
            image__user=self.request.user
        ).order_by('-created_at')[:10]
        return context

@login_required
def recommendations_view(request):
    recommendations = MusicRecommendation.objects.filter(
        image__user=request.user
    ).order_by('-created_at')
    return render(request, 'recommendations.html', {'recommendations': recommendations})

@login_required
def favorites_view(request):
    # You'll need to add a FavoriteSong model or similar to implement this
    return render(request, 'favorites.html', {'favorites': []})

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        try:
            if 'image' not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': 'No image file provided'
                }, status=400)

            image_file = request.FILES['image']
            
            # Create UploadedImage instance
            uploaded_image = UploadedImage.objects.create(
                image=image_file,
                user=request.user
            )
            
            # Process the image and get recommendations
            try:
                recommendations, emotion = process_image_and_get_recommendations(uploaded_image.image.path)
                logger.info(f"Successfully processed image and got {len(recommendations)} recommendations")
                
                # Update the uploaded image with detected emotion
                uploaded_image.detected_emotion = emotion
                uploaded_image.save()
                
                # Save recommendations to database
                saved_recommendations = []
                for rec in recommendations:
                    recommendation = MusicRecommendation.objects.create(
                        image=uploaded_image,
                        song_title=rec['song_title'],
                        artist=rec['artist'],
                        spotify_link=rec.get('spotify_link', ''),
                        preview_url=rec.get('preview_url', ''),
                        emotion=emotion
                    )
                    saved_recommendations.append(recommendation)
                logger.info(f"Saved {len(saved_recommendations)} recommendations to database")
                
                return JsonResponse({
                    'success': True,
                    'image_id': uploaded_image.id,
                    'emotion': emotion,
                    'recommendations': [
                        {
                            'title': rec.song_title,
                            'artist': rec.artist,
                            'preview_url': rec.preview_url,
                            'spotify_link': rec.spotify_link
                        }
                        for rec in saved_recommendations
                    ],
                    'message': f"Successfully detected {emotion} emotion and found {len(recommendations)} songs"
                })
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                uploaded_image.delete()  # Clean up the uploaded image
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                }, status=500)
                
        except Exception as e:
            logger.error(f"Error in upload_image: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=405)

def get_recommendations(request, image_id):
    try:
        image = UploadedImage.objects.get(id=image_id, user=request.user)
        recommendations = MusicRecommendation.objects.filter(image=image)
        
        return JsonResponse({
            'success': True,
            'recommendations': [
                {
                    'title': rec.song_title,
                    'artist': rec.artist,
                    'preview_url': rec.preview_url,
                    'spotify_link': rec.spotify_link
                }
                for rec in recommendations
            ]
        })
        
    except UploadedImage.DoesNotExist:
        return JsonResponse({'error': 'Image not found'}, status=404)
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return JsonResponse({'error': 'Internal server error'}, status=500)

def favorites(request):
    """Handle favorites endpoint"""
    return JsonResponse({
        'success': True,
        'favorites': []  # Return empty list for now
    })
