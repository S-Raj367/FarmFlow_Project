/**
 * Create a URL-friendly slug from a string
 * @param {string} text - The text to slugify
 * @returns {string} - The slugified text
 */
const slugify = (text) => {
    return text
      .toString()
      .toLowerCase()
      .replace(/\s+/g, "-") // Replace spaces with -
      .replace(/[^\w-]+/g, "") // Remove all non-word chars
      .replace(/--+/g, "-") // Replace multiple - with single -
      .replace(/^-+/, "") // Trim - from start of text
      .replace(/-+$/, "") // Trim - from end of text
      .substring(0, 60) // Limit length
  }
  
  module.exports = slugify
  
  