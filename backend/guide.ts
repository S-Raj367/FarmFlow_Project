"use client"

/**
 * FRONTEND INTEGRATION GUIDE
 *
 * This file provides instructions on how to connect the React frontend to this backend.
 *
 * 1. CORS is already configured to allow requests from your frontend (default: http://localhost:3000)
 * 2. Authentication is handled with JWT tokens
 * 3. All API endpoints follow RESTful conventions
 */

// Example of how to connect the frontend to this backend

/**
 * 1. Create an API service in your frontend
 *
 * File: src/services/api.js
 */

/*
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add request interceptor to add auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export default api;
*/

/**
 * 2. Create authentication service
 *
 * File: src/services/auth.service.js
 */

/*
import api from './api';

// Register user
export const register = async (userData) => {
  const response = await api.post('/auth/register', userData);
  if (response.data.token) {
    localStorage.setItem('token', response.data.token);
    localStorage.setItem('user', JSON.stringify(response.data.user));
  }
  return response.data;
};

// Login user
export const login = async (email, password) => {
  const response = await api.post('/auth/login', { email, password });
  if (response.data.token) {
    localStorage.setItem('token', response.data.token);
    localStorage.setItem('user', JSON.stringify(response.data.user));
  }
  return response.data;
};

// Logout user
export const logout = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
};

// Get current user
export const getCurrentUser = () => {
  const user = localStorage.getItem('user');
  return user ? JSON.parse(user) : null;
};

// Check if user is authenticated
export const isAuthenticated = () => {
  return localStorage.getItem('token') !== null;
};
*/

/**
 * 3. Create forum service
 *
 * File: src/services/forum.service.js
 */

/*
import api from './api';

// Get all categories
export const getCategories = async () => {
  const response = await api.get('/categories');
  return response.data;
};

// Get single category with topics
export const getCategory = async (slug) => {
  const response = await api.get(`/categories/${slug}`);
  return response.data;
};

// Get topics with pagination and filters
export const getTopics = async (params) => {
  const response = await api.get('/topics', { params });
  return response.data;
};

// Get single topic with posts
export const getTopic = async (id) => {
  const response = await api.get(`/topics/${id}`);
  return response.data;
};

// Create new topic
export const createTopic = async (topicData) => {
  const response = await api.post('/topics', topicData);
  return response.data;
};

// Create new post or reply
export const createPost = async (postData) => {
  const response = await api.post('/posts', postData);
  return response.data;
};

// Toggle like on topic or post
export const toggleLike = async (data) => {
  const response = await api.post('/likes', data);
  return response.data;
};

// Get user notifications
export const getNotifications = async (params) => {
  const response = await api.get('/notifications', { params });
  return response.data;
};

// Mark notification as read
export const markNotificationAsRead = async (id) => {
  const response = await api.put(`/notifications/${id}`);
  return response.data;
};

// Mark all notifications as read
export const markAllNotificationsAsRead = async () => {
  const response = await api.put('/notifications');
  return response.data;
};
*/

/**
 * 4. Update your AuthDialog component to use these services
 *
 * Example:
 */

/*
import { useState } from "react";
import { login, register } from "@/services/auth.service";

// Inside your component
const handleSignIn = async (e) => {
  e.preventDefault();
  setIsLoading(true);

  try {
    await login(signInData.email, signInData.password);
    toast({
      title: "Success!",
      description: "You have been signed in.",
    });
    onOpenChange(false);
    router.refresh();
  } catch (error) {
    toast({
      title: "Authentication Error",
      description: error.response?.data?.error || "Invalid email or password. Please try again.",
      variant: "destructive",
    });
  } finally {
    setIsLoading(false);
  }
};

const handleRegister = async (e) => {
  e.preventDefault();
  
  // Validate form
  if (registerData.password !== registerData.confirmPassword) {
    toast({
      title: "Passwords do not match",
      description: "Please make sure your passwords match.",
      variant: "destructive",
    });
    return;
  }

  setIsLoading(true);

  try {
    const fullName = `${registerData.firstName} ${registerData.lastName}`.trim();
    
    await register({
      name: fullName,
      email: registerData.email,
      password: registerData.password,
    });
    
    toast({
      title: "Registration Successful",
      description: "Your account has been created and you are now signed in.",
    });
    onOpenChange(false);
    router.refresh();
  } catch (error) {
    toast({
      title: "Registration Error",
      description: error.response?.data?.error || "An error occurred during registration. Please try again.",
      variant: "destructive",
    });
  } finally {
    setIsLoading(false);
  }
};
*/

/**
 * 5. Update your Forum component to use these services
 *
 * Example:
 */

/*
import { useState, useEffect } from "react";
import { getCategories, getTopics } from "@/services/forum.service";

// Inside your component
useEffect(() => {
  const fetchCategories = async () => {
    try {
      const result = await getCategories();
      setCategories(result.data);
    } catch (error) {
      console.error("Error fetching categories:", error);
    }
  };

  const fetchTopics = async () => {
    try {
      const params = {
        page: currentPage,
        limit: 10,
        category: activeCategory !== "all" ? activeCategory : undefined,
        search: searchQuery || undefined,
        sort: currentSort
      };
      
      const result = await getTopics(params);
      setTopics(result.data);
      setPagination(result.pagination);
    } catch (error) {
      console.error("Error fetching topics:", error);
    }
  };

  fetchCategories();
  fetchTopics();
}, [activeCategory, searchQuery, currentPage, currentSort]);
*/

/**
 * 6. Update your Navbar component to show notifications
 *
 * Example:
 */

/*
import { useState, useEffect } from "react";
import { getNotifications } from "@/services/forum.service";

// Inside your component
useEffect(() => {
  // Fetch unread notifications count
  if (isAuthenticated()) {
    const fetchNotifications = async () => {
      try {
        const result = await getNotifications({ unreadOnly: true, limit: 1 });
        setUnreadNotifications(result.unreadCount || 0);
      } catch (error) {
        console.error('Error fetching notifications:', error);
      }
    };

    fetchNotifications();
    
    // Set up polling for notifications
    const interval = setInterval(fetchNotifications, 60000); // Check every minute
    return () => clearInterval(interval);
  }
}, [isAuthenticated]);
*/

