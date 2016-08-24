// Copyright 2015 National ICT Australia Limited (NICTA)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMPONENT_COLLECTION_DOT_H
#define COMPONENT_COLLECTION_DOT_H

#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace Sgt
{
    template<typename T> class ComponentCollection;
    template<typename T> class MutableComponentCollection;

    /// @brief Pointer type to a member of ComponentCollection.
    template<typename T> class ComponentPtr
    {
        friend class ComponentCollection<T>;
        friend class MutableComponentCollection<T>;

        private:
            using Iter = typename std::conditional<
                std::is_const<T>::value,
                typename std::map<std::string, std::shared_ptr<T>>::const_iterator,
                typename std::map<std::string, std::shared_ptr<T>>::iterator>::type;

            /// @brief Constructor for existing component.
            ComponentPtr(const Iter& it) : it_(std::make_unique<Iter>(it)) {}

        public:
            /// @brief Default constructor.
            ComponentPtr() = default;
            
            /// @brief Copy constructor. 
            ComponentPtr(const ComponentPtr& from) : it_(std::make_unique<Iter>(*from.it_)) {}
            
            /// @brief Move constructor. 
            ComponentPtr(ComponentPtr&& from) : it_(std::move(from.it_)) {}

            /// @brief Copy assignment. 
            void operator=(const ComponentPtr& from)
            {
                it_ = std::make_unique<Iter>(*from.it_);
            }
            
            /// @brief Move assignment. 
            void operator=(const ComponentPtr&& from)
            {
                it_ = std::move(from.it_);
            }

            /// @brief Dereference.
            const T& operator*() const {return *((**it_).second);}
            /// @brief Dereference.
            T& operator*() {return *((**it_).second);}

            /// @brief Dereference. 
            const T* operator->() const {return (**it_).second.get();}
            /// @brief Dereference. 
            T* operator->() {return (**it_).second.get();}
            
            /// @brief Converts to shared_ptr. 
            operator std::shared_ptr<const T>() const {return (**it_).second;}
            /// @brief Converts to shared_ptr. 
            operator std::shared_ptr<T>() {return (**it_).second;}

            /// @brief Converts to raw pointer. 
            operator const T*() const {return (**it_).second.get();}
            /// @brief Converts to raw pointer. 
            operator T*() {return (**it_).second.get();}

            /// @brief Cast to shared.
            template<typename U> std::shared_ptr<const U> asShared() const
            {
                return std::dynamic_pointer_cast<const U>((**it_).second);
            }
            /// @brief Cast to shared.
            template<typename U> std::shared_ptr<U> asShared()
            {
                return std::dynamic_pointer_cast<U>((**it_).second);
            }

            /// @brief Cast.
            template<typename U> const U* as() const {return dynamic_cast<const U*>((**it_).second.get());}
            /// @brief Cast.
            template<typename U> U* as() {return dynamic_cast<U*>((**it_).second.get());}
            
            /// @brief Obtain the key in my container.
            const std::string& key() {return (**it_).first;}

            /// @brief Do I refer to an actual component? 
            operator bool() const {return  it_ && (**it_).second;}

            /// @brief Comparison to nullptr.
            bool operator==(const std::nullptr_t& rhs) const {return it_ == nullptr || (**it_).second == nullptr;}
            /// @brief Comparison to nullptr.
            bool operator!=(const std::nullptr_t& rhs) const {return !operator==(rhs);}

            /// @brief Compare two ComponentPtrs.
            ///
            /// Undefined behaviour if they come from different ComponentCollection containers.
            bool operator==(const ComponentPtr& rhs) const
            {
                return (it_ == rhs.it_) || ((**it_).first == (**rhs.it_).first); // Works for nullptr.
            }
            /// @brief Compare two ComponentPtrs.
            ///
            /// Undefined behaviour if they come from different ComponentCollection containers.
            bool operator!=(const ComponentPtr& rhs) const {!operator==(rhs);}

        private:
            typename std::unique_ptr<Iter> it_{nullptr};
    };

    template<typename T> using ConstComponentPtr = ComponentPtr<std::add_const_t<T>>; 

    /// @brief Utility container type combining aspects of map and vector.
    ///
    /// May be iterated over or indexed by a string key or an integer index.
    /// ComponentPtr and ConstComponentPtr are a kind of smart pointer to elements in the container.
    /// They are effectively equivalent to storing an iterator to a map, and as such, are stable under
    /// changes to the mapped type. Thus we can obtain a ComponentPtr, and later replace the value 
    /// stored at the given key with another value, and the ComponentPtr will reference the newly replaced
    /// value.
    template<typename T> class ComponentCollection
    {
        public:
            using ConstPtr = ConstComponentPtr<T>;
            using Ptr = ComponentPtr<T>;

        public:
            ConstPtr operator[](const std::string& key) const {return map_.find(key);}
            Ptr operator[](const std::string& key) {return map_.find(key);}
            
            ConstPtr operator[](size_t idx) const {return vec_[idx];}
            Ptr operator[](size_t idx) {return vec_[idx];}

            size_t size() const {return vec_.size();}

            auto begin() {return vec_.begin();}
            auto end() {return vec_.end();}

            auto begin() const {return vec_.cbegin();}
            auto end() const {return vec_.cend();}
           
            auto cbegin() const {return vec_.cbegin();}
            auto cend() const {return vec_.cend();}
            
            auto rbegin() {return vec_.rbegin();}
            auto rend() {return vec_.rend();}
            
            auto rbegin() const {return vec_.rcbegin();}
            auto rend() const {return vec_.rcend();}

            auto rcbegin() const {return vec_.rcbegin();}
            auto rcend() const {return vec_.rcend();}
            
        protected:
            std::map<std::string, std::shared_ptr<T>> map_;
            std::vector<Ptr> vec_;
    };
    
    template<typename T> class MutableComponentCollection : public ComponentCollection<T>
    {
        public:
            using Ptr = typename ComponentCollection<T>::Ptr;
            using ConstPtr = typename ComponentCollection<T>::ConstPtr;

            Ptr insert(const std::string& key, std::shared_ptr<T> comp)
            {
                auto it = ComponentCollection<T>::map_.find(key);
                if (it == ComponentCollection<T>::map_.end())
                {
                    // Insert new element.
                    it = ComponentCollection<T>::map_.insert(std::make_pair(key, comp)).first;
                    ComponentCollection<T>::vec_.push_back(it);
                }
                else
                {
                    // Replace existing element.
                    it->second = comp;
                }
                return it;
            }

            Ptr reserve(const std::string& key) {insert(key, std::shared_ptr<T>(nullptr));}

            template<typename U, typename ... Args>
            std::pair<Ptr, std::shared_ptr<U>> emplace(const std::string& key, Args&&... args)
            {
                auto comp = std::make_shared<U>(std::forward<Args>(args)...);
                return {insert(key, comp), comp};
            }
            
            std::shared_ptr<T> remove(const std::string& key) 
            {
                std::shared_ptr<T> result;
                auto mapIt = ComponentCollection<T>::map_.find(key);
                if (mapIt != ComponentCollection<T>::map_.end())
                {
                    result = mapIt->second;
                    ComponentCollection<T>::map_.erase(mapIt);
                    auto vecIt = std::find_if(ComponentCollection<T>::vec_.begin(), ComponentCollection<T>::vec_.end(), 
                            [&key](Ptr& p){return p.key() == key;});
                    assert(vecIt != ComponentCollection<T>::vec_.end());
                    ComponentCollection<T>::vec_.erase(vecIt);
                }
                return result;
            }
    };
}

#endif // COMPONENT_COLLECTION_DOT_H
