"use client"

import Link from "next/link"
import { useState, useEffect } from "react"
import { Menu, X, User, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import AuthDialog from "@/components/auth-dialog"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface UserData {
  id: string
  name: string
  email: string
  image: string | null
}

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isAuthOpen, setIsAuthOpen] = useState(false)
  const [user, setUser] = useState<UserData | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check if user is logged in
    const storedUser = localStorage.getItem("user")
    console.log("Stored user:", storedUser) // Debugging

    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser)
        console.log("Parsed user:", parsedUser) // Debugging

        if (parsedUser && parsedUser.name && parsedUser.email) {
          setUser(parsedUser)
        } else {
          console.error("Invalid user data in localStorage")
          localStorage.removeItem("user")
        }
      } catch (error) {
        console.error("Error parsing user data:", error)
        localStorage.removeItem("user")
      }
    }
    setIsLoading(false)
  }, [])

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen)
  }

  const handleSignOut = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("user")
    setUser(null)
    window.location.reload()
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-2xl font-bold text-lime-700">FarmFlow</span>
        </Link>

        {/* Mobile menu button */}
        <Button variant="ghost" className="md:hidden" onClick={toggleMenu}>
          {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </Button>

        {/* Desktop navigation */}
        <nav className="hidden md:flex items-center gap-6">
          <Link href="/" className="text-sm font-medium hover:text-lime-700 transition-colors">
            Home
          </Link>
          <Link href="/crop-recommendation" className="text-sm font-medium hover:text-lime-700 transition-colors">
            Crop Recommendation
          </Link>
          <Link href="/disease-detection" className="text-sm font-medium hover:text-lime-700 transition-colors">
            Disease Detection
          </Link>
          <Link href="/height-estimation" className="text-sm font-medium hover:text-lime-700 transition-colors">
            Height Estimation
          </Link>
          <Link href="/price-prediction" className="text-sm font-medium hover:text-lime-700 transition-colors">
            Price Prediction
          </Link>
          <Link href="/chatbot" className="text-sm font-medium hover:text-lime-700 transition-colors">
            FarmBot Chat
          </Link>
          <Link href="/forum" className="text-sm font-medium hover:text-lime-700 transition-colors">
            Community Forum
          </Link>
        </nav>

        {/* Auth button or user menu */}
        {!isLoading &&
          (user ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                  <Avatar className="h-8 w-8">
                    <AvatarImage src={user?.image || ""} alt={user?.name || "User"} />
                    <AvatarFallback className="bg-lime-100 text-lime-800">
                      {user?.name?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <div className="flex items-center justify-start gap-2 p-2">
                  <div className="flex flex-col space-y-1 leading-none">
                    <p className="font-medium">{user.name}</p>
                    <p className="w-[200px] truncate text-sm text-muted-foreground">{user.email}</p>
                  </div>
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link href="/profile">Profile</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/settings">Settings</Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleSignOut}>
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button
              variant="outline"
              size="sm"
              className="hidden md:flex items-center gap-2"
              onClick={() => setIsAuthOpen(true)}
            >
              <User className="h-4 w-4" />
              Sign In
            </Button>
          ))}
      </div>

      {/* Mobile navigation */}
      <div
        className={cn(
          "container md:hidden overflow-hidden transition-all duration-300 ease-in-out",
          isMenuOpen ? "max-h-96" : "max-h-0",
        )}
      >
        <nav className="flex flex-col space-y-4 py-4">
          <Link
            href="/"
            className="text-sm font-medium hover:text-lime-700 transition-colors"
            onClick={() => setIsMenuOpen(false)}
          >
            Home
          </Link>
          <Link
            href="/crop-recommendation"
            className="text-sm font-medium hover:text-lime-700 transition-colors"
            onClick={() => setIsMenuOpen(false)}
          >
            Crop Recommendation
          </Link>
          <Link
            href="/disease-detection"
            className="text-sm font-medium hover:text-lime-700 transition-colors"
            onClick={() => setIsMenuOpen(false)}
          >
            Disease Detection
          </Link>
          <Link
            href="/height-estimation"
            className="text-sm font-medium hover:text-lime-700 transition-colors"
            onClick={() => setIsMenuOpen(false)}
          >
            Height Estimation
          </Link>
          <Link
            href="/price-prediction"
            className="text-sm font-medium hover:text-lime-700 transition-colors"
            onClick={() => setIsMenuOpen(false)}
          >
            Price Prediction
          </Link>
          <Link
            href="/chatbot"
            className="text-sm font-medium hover:text-lime-700 transition-colors"
            onClick={() => setIsMenuOpen(false)}
          >
            FarmBot Chat
          </Link>
          <Link
            href="/forum"
            className="text-sm font-medium hover:text-lime-700 transition-colors"
            onClick={() => setIsMenuOpen(false)}
          >
            Community Forum
          </Link>

          {!isLoading &&
            (user ? (
              <>
                <div className="flex items-center gap-2 py-2">
                  <Avatar className="h-8 w-8">
                    <AvatarImage src={user?.image || ""} alt={user?.name || "User"} />
                    <AvatarFallback className="bg-lime-100 text-lime-800">
                      {user?.name?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">{user.name}</span>
                    <span className="text-xs text-gray-500">{user.email}</span>
                  </div>
                </div>
                <Link
                  href="/profile"
                  className="text-sm font-medium hover:text-lime-700 transition-colors pl-2"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Profile
                </Link>
                <Link
                  href="/settings"
                  className="text-sm font-medium hover:text-lime-700 transition-colors pl-2"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Settings
                </Link>
                <Button
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2 justify-start pl-2"
                  onClick={() => {
                    handleSignOut()
                    setIsMenuOpen(false)
                  }}
                >
                  <LogOut className="h-4 w-4" />
                  Log out
                </Button>
              </>
            ) : (
              <Button
                variant="outline"
                size="sm"
                className="flex items-center gap-2 w-full justify-center"
                onClick={() => {
                  setIsMenuOpen(false)
                  setIsAuthOpen(true)
                }}
              >
                <User className="h-4 w-4" />
                Sign In
              </Button>
            ))}
        </nav>
      </div>

      {/* Auth Dialog */}
      <AuthDialog open={isAuthOpen} onOpenChange={setIsAuthOpen} />
    </header>
  )
} 