"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { isAuthenticated, getCurrentUser } from "@/lib/auth-helper"

interface UserData {
  id: string
  name: string
  email: string
  image: string | null
  createdAt: string
}

export default function ProfilePage() {
  const [user, setUser] = useState<UserData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    // Check if user is authenticated
    if (!isAuthenticated()) {
      router.push("/")
      return
    }

    // Get user data
    const userData = getCurrentUser()
    if (userData) {
      setUser(userData as UserData)
    }

    setIsLoading(false)
  }, [router])

  if (isLoading) {
    return (
      <div className="container py-10">
        <div className="flex justify-center items-center min-h-[50vh]">
          <p>Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="container py-10">
        <div className="flex justify-center items-center min-h-[50vh]">
          <Card>
            <CardHeader>
              <CardTitle>Not Authenticated</CardTitle>
              <CardDescription>You need to sign in to view this page</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={() => router.push("/")}>Go to Home</Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="container py-10">
      <div className="max-w-3xl mx-auto">
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-2xl">Profile</CardTitle>
            <CardDescription>View and manage your profile information</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-6 items-start">
              <div className="flex flex-col items-center gap-2">
                <Avatar className="h-24 w-24">
                  <AvatarImage src={user.image || ""} alt={user.name} />
                  <AvatarFallback className="text-2xl bg-lime-100 text-lime-800">{user.name.charAt(0)}</AvatarFallback>
                </Avatar>
                <Button variant="outline" size="sm">
                  Change Avatar
                </Button>
              </div>

              <div className="flex-1 space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-500">Name</h3>
                  <p className="text-lg">{user.name}</p>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-gray-500">Email</h3>
                  <p className="text-lg">{user.email}</p>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-gray-500">Member Since</h3>
                  <p className="text-lg">{new Date(user.createdAt).toLocaleDateString()}</p>
                </div>

                <div className="pt-4">
                  <Button className="bg-lime-700 hover:bg-lime-800">Edit Profile</Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

