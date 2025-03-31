import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors'
import morgan from 'morgan'
import connectDB from "./config/db.js"

dotenv.config();

// Import routes
import authRoutes from "./routes/auth.routes.js"

// Initialize express app
const app = express()

// Connect to MongoDB
connectDB()

// Middleware
app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    credentials: true,
  }),
)
app.use(express.json())
app.use(morgan("dev"))

// Routes
app.use("/api/auth", authRoutes)

// Health check route
app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok", message: "Auth server is running" })
})

// Error handler
app.use((err, req, res, next) => {
  console.error(err.stack)
  res.status(err.statusCode || 500).json({
    success: false,
    error: err.message || "Server Error",
  })
})

// Start server
const PORT = process.env.PORT || 5001
app.listen(PORT, () => {
  console.log(`Auth server running on port ${PORT}`)
})

