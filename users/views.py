from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from django.db import transaction
from .models import UserProfile

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        favorite_genres = request.POST.get('favorite_genres', '')
        profile_picture = request.FILES.get('profile_picture')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return redirect('register')
        
        try:
            with transaction.atomic():
                user = User.objects.create_user(username=username, password=password, email=email)
                profile = UserProfile.objects.get_or_create(user=user)[0]
                
                if profile_picture:
                    # Validate file type
                    if not profile_picture.content_type.startswith('image/'):
                        raise ValueError('Invalid file type. Please upload an image file.')
                    profile.profile_picture = profile_picture
                
                if favorite_genres:
                    profile.favorite_genres = favorite_genres
                
                profile.save()
                login(request, user)
            return redirect('home')
        except ValueError as ve:
            messages.error(request, str(ve))
            return redirect('register')
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
            return redirect('register')
    return render(request, 'users/register.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid credentials')
    return render(request, 'users/login.html')

def logout_view(request):
    logout(request)
    return redirect('login') 