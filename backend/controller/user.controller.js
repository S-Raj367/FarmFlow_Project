import User from "../models/User"
import ErrorResponse from "../utils/errorResponse"

// @desc    Get all users
// @route   GET /api/users
// @access  Private/Admin
exports.getUsers = async (req, res, next) => {
  try {
    const users = await User.find()

    res.status(200).json({
      success: true,
      count: users.length,
      data: users,
    })
  } catch (err) {
    next(err)
  }
}

// @desc    Get single user
// @route   GET /api/users/:id
// @access  Private
exports.getUser = async (req, res, next) => {
  try {
    const user = await User.findById(req.params.id)

    if (!user) {
      return next(new ErrorResponse(`User not found with id of ${req.params.id}`, 404))
    }

    // Get user's recent activity
    const recentTopics = await user
      .model("Topic")
      .find({ author: user._id })
      .sort({ createdAt: -1 })
      .limit(5)
      .populate("category")

    const recentPosts = await user
      .model("Post")
      .find({ author: user._id, parent: null })
      .sort({ createdAt: -1 })
      .limit(5)
      .populate("topic")

    // Get counts
    const topicsCount = await user.model("Topic").countDocuments({ author: user._id })
    const postsCount = await user.model("Post").countDocuments({ author: user._id })

    res.status(200).json({
      success: true,
      data: {
        user,
        recentTopics,
        recentPosts,
        counts: {
          topics: topicsCount,
          posts: postsCount,
        },
      },
    })
  } catch (err) {
    next(err)
  }
}

// @desc    Update user
// @route   PUT /api/users/:id
// @access  Private
exports.updateUser = async (req, res, next) => {
  try {
    let user = await User.findById(req.params.id)

    if (!user) {
      return next(new ErrorResponse(`User not found with id of ${req.params.id}`, 404))
    }

    // Make sure user is updating their own profile or is an admin
    if (req.user.id !== req.params.id && req.user.role !== "admin") {
      return next(new ErrorResponse(`User ${req.user.id} is not authorized to update this profile`, 403))
    }

    // Fields to update
    const fieldsToUpdate = {
      name: req.body.name,
      bio: req.body.bio,
      location: req.body.location,
      image: req.body.image,
    }

    // Remove undefined fields
    Object.keys(fieldsToUpdate).forEach((key) => fieldsToUpdate[key] === undefined && delete fieldsToUpdate[key])

    user = await User.findByIdAndUpdate(req.params.id, fieldsToUpdate, {
      new: true,
      runValidators: true,
    })

    res.status(200).json({
      success: true,
      data: user,
    })
  } catch (err) {
    next(err)
  }
}

// @desc    Delete user
// @route   DELETE /api/users/:id
// @access  Private/Admin
exports.deleteUser = async (req, res, next) => {
  try {
    const user = await User.findById(req.params.id)

    if (!user) {
      return next(new ErrorResponse(`User not found with id of ${req.params.id}`, 404))
    }

    await user.remove()

    res.status(200).json({
      success: true,
      data: {},
    })
  } catch (err) {
    next(err)
  }
}

