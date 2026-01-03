import express from 'express';
const router = express.Router()
import { getUsers, getUser, updateUser, deleteUser } from "../controllers/user.controller"
import { protect, authorize }  from "../middleware/auth"

// All routes below this are protected and require authentication
router.use(protect)

// Routes
router.route("/").get(authorize("admin"), getUsers)

router.route("/:id").get(getUser).put(updateUser).delete(authorize("admin"), deleteUser)

export default router