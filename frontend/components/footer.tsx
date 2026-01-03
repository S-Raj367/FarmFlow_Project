import Link from "next/link"
import { Facebook, Instagram, Twitter } from "lucide-react"

export default function Footer() {
  return (
    <footer className="bg-lime-50 border-t">
      <div className="container py-8 md:py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-4 text-lime-700">FarmFlow</h3>
            <p className="text-sm text-gray-600">
              Modern agricultural solutions for smart farming and sustainable growth.
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-4 text-lime-700">Features</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/crop-recommendation" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Crop Recommendation
                </Link>
              </li>
              <li>
                <Link href="/disease-detection" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Disease Detection
                </Link>
              </li>
              <li>
                <Link href="/height-estimation" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Height Estimation
                </Link>
              </li>
              <li>
                <Link href="/price-prediction" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Price Prediction
                </Link>
              </li>
              <li>
                <Link href="/chatbot" className="text-gray-600 hover:text-lime-700 transition-colors">
                  FarmBot Chat
                </Link>
              </li>
              <li>
                <Link href="/forum" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Community Forum
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-4 text-lime-700">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/blog" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Blog
                </Link>
              </li>
              <li>
                <Link href="/guides" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Guides
                </Link>
              </li>
              <li>
                <Link href="/research" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Research
                </Link>
              </li>
              <li>
                <Link href="/faq" className="text-gray-600 hover:text-lime-700 transition-colors">
                  FAQ
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-4 text-lime-700">Company</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/about" className="text-gray-600 hover:text-lime-700 transition-colors">
                  About Us
                </Link>
              </li>
              <li>
                <Link href="/contact" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Contact
                </Link>
              </li>
              <li>
                <Link href="/privacy" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/terms" className="text-gray-600 hover:text-lime-700 transition-colors">
                  Terms of Service
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-gray-200 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-gray-600">© {new Date().getFullYear()} FarmFlow. All rights reserved.</p>
          <div className="flex space-x-4 mt-4 md:mt-0">
            <Link href="#" className="text-gray-600 hover:text-lime-700 transition-colors">
              <Facebook className="h-5 w-5" />
              <span className="sr-only">Facebook</span>
            </Link>
            <Link href="#" className="text-gray-600 hover:text-lime-700 transition-colors">
              <Twitter className="h-5 w-5" />
              <span className="sr-only">Twitter</span>
            </Link>
            <Link href="#" className="text-gray-600 hover:text-lime-700 transition-colors">
              <Instagram className="h-5 w-5" />
              <span className="sr-only">Instagram</span>
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}

