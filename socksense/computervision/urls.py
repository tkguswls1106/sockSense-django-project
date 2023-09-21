from django.urls import path

from . import views

urlpatterns = [
    path('pair/', views.similarity, name='pair_similarity'),
    path('styling/', views.sockColor, name='styling_sockColor'),
]