// Helper functions for authentication

// Check if user is authenticated
export const isAuthenticated = () => {
    if (typeof window === "undefined") return false
  
    return localStorage.getItem("token") !== null
  }
  
  // Get current user
  export const getCurrentUser = () => {
    if (typeof window === "undefined") return null
  
    const user = localStorage.getItem("user")
    return user ? JSON.parse(user) : null
  }
  
  // Get auth token
  export const getToken = () => {
    if (typeof window === "undefined") return null
  
    return localStorage.getItem("token")
  }
  
  // Logout user
  export const logout = () => {
    if (typeof window === "undefined") return
  
    localStorage.removeItem("token")
    localStorage.removeItem("user")
  }
  
  // Fetch current user from API
  export const fetchCurrentUser = async () => {
    const token = getToken()
    if (!token) return null
  
    try {
      const response = await fetch("http://localhost:5000/api/auth/me", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
  
      if (!response.ok) {
        throw new Error("Failed to fetch user")
      }
  
      const data = await response.json()
      return data.user
    } catch (error) {
      console.error("Error fetching current user:", error)
      return null
    }
  }
  
  